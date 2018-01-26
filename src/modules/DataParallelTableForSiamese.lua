--[[
   This file implements data parallelism for Torch modules.

   The same model is replicated on multiple GPUs. The input is split, typically
   into smaller mini-batches. Each replicated model handles only its portion of the input.
   The weight updates for each replica are summed together on the first replica
   in accGradParameters.

   By default, this module uses only one thread and relies on asynchronous kernel launches.
   To use multiple threads, call DataParallelTableForSiamese:threads(initFunc).

   For best performance, install NCCL:
    https://github.com/NVIDIA/nccl
    https://github.com/ngimel/nccl.torch
]]--
require 'nn'
local DataParallelTableForSiamese, parent = torch.class('nn.DataParallelTableForSiamese', 'nn.Container')

local Impls = {}
local BasicImpl = torch.class('nn.DataParallelTableForSiamese.Basic', Impls)
local ThreadsImpl = torch.class('nn.DataParallelTableForSiamese.Threads', Impls)


-- NCCL does not work when CUDA_LAUNCH_BLOCKING is set
local cudaLaunchBlocking = os.getenv('CUDA_LAUNCH_BLOCKING') == '1'

-- extracts the value at idx from each entry in tbl
local function pluck(tbl, idx)
   local r = {}
   for n, val in ipairs(tbl) do
      r[n] = val[idx]
   end
   return r
end

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
   local stream = cutorch.getStream()
   if stream ~= 0 then
      cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
   end
end

function DataParallelTableForSiamese:__init(dimension, flattenParams, usenccl)
   parent.__init(self)
   if not dimension then
      error "must specify a dimension!"
   end
   
   self.dimension = dimension
   self.modules = {}
   self.gpuAssignments = {}  -- Which gpuid each module sits on
   self.inputGpu = {}  -- inputs for each gpu
   self.gradOutputGpu = {} -- gradOutputs for each gpu
   self.outputGpu = {} -- outputs for each gpu
   self.gradInputGpu = {} -- gradInput for each gpu
   self.flattenedParams = nil -- flattened parameters for each gpu
   self.flattenParams = flattenParams or false
   self.usenccl = false
   self.needsSync = false
   self.impl = Impls.Basic(self)
   if usenccl then
      assert(self.flattenParams, 'cannot use nccl without flattenParams')
      self.usenccl = pcall(require, 'nccl')
      if not self.usenccl then
         print("warning: could not load nccl, falling back to default communication")
      end
   end
end

function DataParallelTableForSiamese:add(module, gpus)
   if type(gpus) == 'number' then
      if #self.modules == 0 then
         table.insert(self.modules, module)
      end
      table.insert(self.gpuAssignments, gpus)
      return self
   end

   assert(torch.type(gpus) == 'table' and #gpus >= 1, 'table of GPU IDs required')
   assert(#self.modules == 0, 'add should only be called once with a table of GPU assignments')
   self.modules[1] = module
   self.gpuAssignments = gpus
   return self
end

function DataParallelTableForSiamese:threads(initFunc)
   require 'threads'
   self.impl:close()
   self.impl = Impls.Threads(self, initFunc)
   return self
end

function DataParallelTableForSiamese:__tostring()
   return 'DataParallelTableForSiamese: ' .. #self.gpuAssignments .. ' x ' .. tostring(self.modules[1])
end

function DataParallelTableForSiamese:get(index)
   return self.modules[index]
end

-- this flattens parameters, so that syncParameters and accGradParameters can be much more efficient
function DataParallelTableForSiamese:flattenParameters()
   self.flattenedParams = self.impl:exec(function(module)
      local p, dp = module:parameters()
      local flattened = true
      for i=2,#p do
         if p[i]:storage() ~= p[1]:storage() 
            or dp[i]:storage() ~= dp[1]:storage() then
            flattened = false
            break
         end
      end
      if flattened then
         local pp = torch.CudaTensor(p[1]:storage(), p[1]:storageOffset(), 
                    p[#p]:storageOffset()+p[#p]:numel()-1)
         local dpp = torch.CudaTensor(dp[1]:storage(), dp[1]:storageOffset(), 
                     dp[#dp]:storageOffset()+dp[#dp]:numel()-1)
         return {pp, dpp}
      else
         return { module:getParameters() }
      end
   end)
   self.flattenParams = true
end

function DataParallelTableForSiamese:getParameters()
   self:flattenParameters()
   return table.unpack(self.flattenedParams[1])
end

local function hasFlattenedParmeters(self)
   if not self.flattenedParams then
      return false
   end
   for _, param in ipairs(self.modules[1]:parameters()) do
      if param:storage() ~= self.flattenedParams[1][1]:storage() then
         return false
      end
   end
   return true
end

function DataParallelTableForSiamese:training()
   self.impl:exec(function(module)
      module:training()
   end)
   parent.training(self)
end

function DataParallelTableForSiamese:evaluate()
   self.impl:exec(function(module)
      module:evaluate()
   end)
   parent.evaluate(self)
end

function DataParallelTableForSiamese:updateOutput(input)
   if self.flattenParams and not hasFlattenedParmeters(self) then
      self:flattenParameters()
   end
   if self.needsSync then
      self:syncParameters()
   end

   local prevGpuid = cutorch.getDevice()

   -- distribute the input to GPUs
   self:_distribute(self.inputGpu, input)

   -- update output for each module
   local inputGpu = self.inputGpu
   self.outputGpu = self.impl:exec(function(m, i)
      if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
         return torch.CudaTensor()
      elseif (not torch.isTensor(inputGpu[i])) and (table.getn(inputGpu[i]) == 0) then
         return torch.CudaTensor()        
      else
        return m:updateOutput(inputGpu[i])
      end
   end)
   
   -- concatenate the outputs to the base GPU
   for i, s in ipairs(self.outputGpu) do
     local currentOutput = self.outputGpu[i]
     if(currentOutput:nDimension() == 1) then self.outputGpu[i] = currentOutput:view(1, currentOutput:size(1)); end
   end
   
   self.output = self:_concat(self.output, self.outputGpu)
   cutorch.setDevice(prevGpuid)

   return self.output
end

function DataParallelTableForSiamese:moduleParameters()
   -- Returns a table containing the parameters for each replica
   if self.flattenedParams then
      local res = {}
      for i, params in ipairs(self.flattenedParams) do
         res[i] = { {params[1]}, {params[2]} }
      end
      return res
   end
   return self.impl:exec(function(m)
      return { m:parameters() }
   end)
end

function DataParallelTableForSiamese:__backward(method, input, gradOutput, scale)
   local prevGpuid = cutorch.getDevice()
   local inputGpu, gradOutputGpu = self.inputGpu, self.gradOutputGpu

   if method == 'backward' or method == 'updateGradInput' then
      -- distribute the gradOutput to GPUs
      self:_distribute(self.gradOutputGpu, gradOutput)

      self.gradInputGpu = self.impl:exec(function(m, i)
         if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
         elseif (not torch.isTensor(inputGpu[i])) and (table.getn(inputGpu[i]) == 0) then
            return {}
         else
            return m[method](m, inputGpu[i], gradOutputGpu[i], scale)
         end
      end)

     -- if self.gradInput then
         -- concatenate the gradInput to the base GPU
         --self.gradInput = self:_concat(self.gradInput, self.gradInputGpu)
      --end
   end

   if method == 'accGradParameters' then
      self.impl:exec(function(m, i)
         if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
         elseif (not torch.isTensor(inputGpu[i])) and (table.getn(inputGpu[i]) == 0) then
            return torch.CudaTensor()            
         else
            return m:accGradParameters(inputGpu[i], gradOutputGpu[i], scale)
         end
      end)
   end

   if method == 'backward' or method == 'accGradParameters' then
      local params = self:moduleParameters()
      -- Accumulate the gradients onto the base GPU
      if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
         if #self.gpuAssignments > 1 then
            nccl.reduce(pluck(self.flattenedParams, 2), nil, true, 1)
         end
      else
         self:_reduce(pluck(params, 2))
      end
      -- Zero out gradients on the other GPUs
      for i = 2, #self.gpuAssignments do
         cutorch.setDevice(self.gpuAssignments[i])
         for _, gradParam in ipairs(params[i][2]) do
            gradParam:zero()
         end
      end
      self.needsSync = true
   end

   cutorch.setDevice(prevGpuid)
   return self.gradInput
end

function DataParallelTableForSiamese:backward(input, gradOutput, scale)
   return self:__backward('backward', input, gradOutput, scale)
end

function DataParallelTableForSiamese:updateGradInput(input, gradOutput)
   return self:__backward('updateGradInput', input, gradOutput)
end

function DataParallelTableForSiamese:accGradParameters(input, gradOutput, scale)
   self:__backward('accGradParameters', input, gradOutput, scale)
end

function DataParallelTableForSiamese:syncParameters()
   local prevGpuid = cutorch.getDevice()
   if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
      if #self.gpuAssignments > 1 then
         nccl.bcast(pluck(self.flattenedParams, 1), true, 1)
      end
   else
      self:_broadcast(pluck(self:moduleParameters(), 1))
   end
   self.needsSync = false
   cutorch.setDevice(prevGpuid)
end

function DataParallelTableForSiamese:accUpdateGradParameters(input, gradOutput, lr)
   error("accUpdateGradParameters not supported for DataParallelTableForSiamese.")
end

function DataParallelTableForSiamese:zeroGradParameters()
   local prevGpuid = cutorch.getDevice()
   if self.flattenedParams then
      for i, parameters in ipairs(self.flattenedParams) do
         cutorch.setDevice(self.gpuAssignments[i])
         parameters[2]:zero()
      end
   else
      self.impl:exec(function(m)
         m:zeroGradParameters()
      end)
   end
   cutorch.setDevice(prevGpuid)
end

function DataParallelTableForSiamese:updateParameters(learningRate)
   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:updateParameters(learningRate)
   self:syncParameters()
   cutorch.setDevice(prevGpuid)
end

function DataParallelTableForSiamese:parameters()
   return self.modules[1]:parameters()
end

function DataParallelTableForSiamese:share(mlp,...)
   error("Share not supported for DataParallelTableForSiamese")
end

function DataParallelTableForSiamese:clone(...)
   assert(select('#',...) == 0, "Sharing not supported for DataParallelTableForSiamese")
   return parent.clone(self)
end

function DataParallelTableForSiamese:reset(stdv)
   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:reset(stdv)
   self:syncParameters()
   cutorch.setDevice(prevGpuid)
end

function DataParallelTableForSiamese:type(typeStr)
   assert(typeStr == 'torch.CudaTensor', 'DataParallelTableForSiamese supports only torch.CudaTensor type')
   for i, m in ipairs(self.modules) do
      m:type(typeStr)
   end
   return self
end

-- Backward compatibility purposes
DataParallelTableForSiamese.__version = 3

-- DataParallelTableForSiamese.deserializeNGPUs controls how many GPUs to deserialize
-- upon, otherwise will deserialize to as many GPUs as serialized and error
-- out if it doesn;t have enough available
function DataParallelTableForSiamese:__read(file, version)
   if version < 2 then
      local var = file:readObject()
      for k, v in pairs(var) do
         self[k] = v
      end
      self.impl = self.impl or Impls.Basic(self)
      return
   end

   -- Pre-read gpuAssignments and either use them of ignore them depending on
   -- whether DataParallelTableForSiamese.deserializeNGPUs is set.
   local gpuAssignments = file:readObject()
   if DataParallelTableForSiamese.deserializeNGPUs then
      gpuAssignments = {}
      for i = 1, DataParallelTableForSiamese.deserializeNGPUs do gpuAssignments[i] = i end
      if DataParallelTableForSiamese.deserializeNGPUs > cutorch.getDeviceCount() then
         error('Deserialization requested on too many GPUs: ' ..
                  DataParallelTableForSiamese.deserializeNGPUs .. ' vs ' ..
                  cutorch.getDeviceCount() .. ' available')
      end
   end

   -- If DataParallelTableForSiamese.deserializeNGPUs, deserialization overrides
   -- gpu assignments anyway. If not, we need as many GPUs as the max,
   -- there may be holes.
   local nGPUs = math.max(unpack(gpuAssignments))
   if nGPUs > cutorch.getDeviceCount() then
      error('Model was serialized on ' ..
               math.max(unpack(gpuAssignments)) ..
               ' nGPUs, but you are running on ' .. cutorch.getDeviceCount() ..
               ' please set DataParallelTableForSiamese.deserializeNGPUs to ignore ' ..
               ' serialized tower-GPU assignments')
   end

   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(gpuAssignments[1])
   -- Deserialize from table
   local var = file:readObject()
   for k, v in pairs(var) do
      self[k] = v
   end
   cutorch.setDevice(prevGpuid)

   if self.usenccl then
      self.usenccl = pcall(require, 'nccl')
   end
   if not self.impl then
      self.impl = Impls.Basic(self)
   end

   -- use previously deserialize / recomputed gpuAssignments
   self.gpuAssignments = gpuAssignments
   assert(#self.modules == 1)

   local flattenedParams = self.flattenedParams
   if flattenedParams then
      self.flattenedParams = self.impl:exec(function(m, i)
         if i == 1 then
            return flattenedParams[1]
         else
            return { m:getParameters() }
         end
      end)
   end
end

function DataParallelTableForSiamese:__write(file)
   -- Prewrite the current assignments, we may need them to
   -- deserialize the first tower
   file:writeObject(self.gpuAssignments)
   -- Convert to table
   local t = {}
   for k, v in pairs(self) do
      -- Only keep the flattenedParams from the first module
      if k  == 'flattenedParams' then
         t[k] = {v[1]}
      elseif k == 'inputGpu' or k == 'outputGpu' or k == 'gradInputGpu' or k == 'gradOutputGpu' then
         t[k] = {}
      elseif k == 'buffer' then
         t[k] = nil
      else
         t[k] = v
      end
   end
   file:writeObject(t)
   -- Force synchronization, this keeps you honest
   self:syncParameters()
end

function DataParallelTableForSiamese:apply(callback)
   parent.apply(self, callback)
   self.impl:applyChanges()
end

local function sliceRange(nElem, idx, splits)
    -- calculate the count of common elements for all the GPUs
   local commonEltsPerMod = math.floor(nElem / splits)
   
    -- calculate the count of reamining elements for which the element-count shall be commonEltsPerMod + 1
   local remainingElts = nElem - (commonEltsPerMod * splits)
   
   -- according to current idx, how much "commonEltsPerMod + 1" elements are there?
   local commonPlusOneEltsCount = math.min(idx - 1, remainingElts)
   -- according to current idx, how much "commonEltsPerMod" elements are there?
   local commonEltsCount = (idx - 1) - commonPlusOneEltsCount 
   
   -- determine the start index
   local rangeStart = (commonPlusOneEltsCount * (commonEltsPerMod + 1)) + 
                        (commonEltsCount * commonEltsPerMod) + 1
                        
    -- determine the total elements for current index
   local currentElts = commonEltsPerMod
   if(idx <= remainingElts) then currentElts = commonEltsPerMod + 1 end

    -- return start index and elements count
   return rangeStart, currentElts
end

local function sumSizes(tensors, dim)
   local size
   for i=1,#tensors do
      if tensors[i]:numel() > 0 then
         if size then
            size[dim] = size[dim] + tensors[i]:size(dim)
         else
            size = tensors[i]:size()
         end
      end
   end
   return size
end

-- Copies the parameters from the first replica to all other replicas
function DataParallelTableForSiamese:_broadcast(params)
   for moduleIdx = 2, #params do
      for paramIdx = 1, #params[moduleIdx] do
         params[moduleIdx][paramIdx]:copy(params[1][paramIdx])
      end
      waitForDevice(self.gpuAssignments[moduleIdx], self.gpuAssignments[1])
   end
end

-- Sums all the gradParams on to the first replica
function DataParallelTableForSiamese:_reduce(gradParams)
   local dstGpuid = self.gpuAssignments[1]
   cutorch.setDevice(dstGpuid)

   self.buffer = self.buffer or torch.CudaTensor()
   for moduleIdx = 2, #gradParams do
      for paramIdx = 1, #gradParams[moduleIdx] do
         local dst = gradParams[1][paramIdx]
         local src = gradParams[moduleIdx][paramIdx]

         -- Synchronize before and after copy to ensure that it doesn't overlap
         -- with this add or previous adds
         waitForDevice(self.gpuAssignments[moduleIdx], dstGpuid)
         self.buffer:resizeAs(src):copy(src)
         waitForDevice(dstGpuid, self.gpuAssignments[moduleIdx])

         dst:add(self.buffer)
      end
   end
end

function DataParallelTableForSiamese:_distribute(dst, src)
   -- determine total size
   for i = 1, #self.gpuAssignments do
      cutorch.setDevice(self.gpuAssignments[i])
      dst[i] = self:_distributeTensorRecursive(dst[i], src, i, #self.gpuAssignments)
   end
end

-- _distributeTensorRecursive - if the src is a tensor then the function slices
-- it long self.dimension and copies each portion into each child module.
-- Otherwise it does a recursive call on tables.
function DataParallelTableForSiamese:_distributeTensorRecursive(dst, src, idx, n)
   if torch.type(src) == 'table' then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- hold on to current input
      local currentInput = src[idx]
      
      -- Recurse on the table
      if(currentInput ~= nil) then
        dst = currentInput
        --for i = 1, table.getn(currentInput) do
        --   dst[i] = currentInput[i]:clone()
        --end
      else
        dst = {}
      end
      return dst
   end

   assert(torch.isTensor(src), 'input must be a tensor or table of tensors')
   assert(src:type() == 'torch.CudaTensor' or src:type() == 'torch.FloatTensor',
      'input must be a CUDA or Float tensor')

   dst = torch.type(dst) == 'torch.CudaTensor' and dst or torch.CudaTensor()

   local index, size = sliceRange(src:size(self.dimension), idx, n)
   if size == 0 then
      dst:resize(0)
   else
      local slice = src:narrow(self.dimension, index, size)
      
      --squeeze the dimension only if the slice is of more than 2 dimensions
      if(slice:numel() > 1) then
        slice = slice:squeeze() --remove 1D dimensions
      else
        slice = slice:squeeze(1)
      end
      
      dst:resize(slice:size()):copyAsync(slice)
      if slice.getDevice then
         waitForDevice(dst:getDevice(), slice:getDevice())
      end
   end

   return dst
end

-- _concat - if the src is a tensor then the function copies it
-- into the dst slice along self.dimension.
-- Otherwise it does a recursive call on tables.
function DataParallelTableForSiamese:_concat(dst, src)
   dst = self:_concatTensorRecursive(dst, src)
   for i=2,#self.gpuAssignments do
      waitForDevice(self.gpuAssignments[1], self.gpuAssignments[i])
   end
   return dst
end

function DataParallelTableForSiamese:_concatTensorRecursive(dst, src)
   if torch.type(src[1]) == 'table' then
      if torch.type(dst) ~= 'table' or #src[1] ~= #dst then
         dst = {}
      end
      
      cutorch.setDevice(self.gpuAssignments[1])
      
      --just copy the tensors in table to current GPU
      for i, s in ipairs(src) do
         portedGradInput = {}
         
         for j, internalTensor in ipairs(s) do
            portedGradInput[j] = internalTensor:clone()
         end
         
         dst[i] = portedGradInput
      end
      return dst
   end

   assert(torch.isTensor(src[1]), 'input must be a tensor or table of tensors')

   cutorch.setDevice(self.gpuAssignments[1])
   dst = torch.type(dst) == 'torch.CudaTensor' and dst or torch.CudaTensor()

   local cumsum = sumSizes(src, self.dimension)
   
   if cumsum == nil then return dst end

   dst:resize(cumsum)

   local start = 1

   for i, s in ipairs(src) do
      if torch.numel(s) > 0 then
         local sz = s:size(self.dimension)
         dst:narrow(self.dimension, start, sz):copy(s)
         start = start + sz
      end
   end

   return dst
end

-- Single-thread dispatch
function BasicImpl:__init(dpt)
   self.dpt = dpt
end

-- Re-copies the first replica onto all the other GPUs, if already setup
function BasicImpl:applyChanges()
   if self.modules then
      local prevGpuid = cutorch.getDevice()
      self.modules = { self.dpt.modules[1] }
      collectgarbage()
      for i=2,#self.dpt.gpuAssignments do
         cutorch.setDevice(self.dpt.gpuAssignments[i])
         table.insert(self.modules, self.dpt.modules[1]:clone())
      end
      cutorch.setDevice(prevGpuid)
   end
end

-- Copies the first replica onto all the other GPUs, if necessary
function BasicImpl:setup()
   if not self.modules then
      self.modules = {}
      self:applyChanges()
   end
end

-- Applies a function to each replica, combining the results into a table
function BasicImpl:exec(closure)
   local prevGpuid = cutorch.getDevice()
   self:setup()
   local res = {}
   for i, gpu in ipairs(self.dpt.gpuAssignments) do
      cutorch.setDevice(gpu)
      res[i] = closure(self.modules[i], i)
   end
   cutorch.setDevice(prevGpuid)
   return res
end

function BasicImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= 'modules' then
         t[k] = v
      end
   end
   file:writeObject(t)
end

function BasicImpl:close()
   self.modules = nil
end

-- Multi-threaded dispatch
function ThreadsImpl:__init(dpt, initFunc)
   self.dpt = dpt
   self.initFunc = initFunc
end

function ThreadsImpl:applyChanges()
   if self.__threads then
      local module = self.dpt.modules[1]
      for i, gpu in ipairs(self.dpt.gpuAssignments) do
         self.__threads:addjob(i, function()
            cutorch.setDevice(gpu)
            if i == 1 then
               _G.module = module
            else
               _G.module = nil
               collectgarbage()
               _G.module = module:clone()
            end
         end)
      end
      self.__threads:synchronize()
   end
end

function ThreadsImpl:setup()
   if not self.__threads then
      local threads = require 'threads'
      threads.Threads.serialization('threads.sharedserialize')
      self.__threads = threads.Threads(
         #self.dpt.gpuAssignments,
         function() require 'cunn' end,
         self.initFunc)
      self.__threads:specific(true)
      self:applyChanges()
   end
end

function ThreadsImpl:exec(closure)
   self:setup()
   local res = {}
   for i=1,#self.dpt.gpuAssignments do
      self.__threads:addjob(i,
         function()
            return closure(_G.module, i)
         end,
         function (_res_)
            res[i] = _res_
         end)
   end
   self.__threads:synchronize()
   return res
end

function ThreadsImpl:close()
   self.__threads:terminate()
   self.__threads = nil
end

function ThreadsImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= '__threads' then
         t[k] = v
      end
   end
   file:writeObject(t)
end
