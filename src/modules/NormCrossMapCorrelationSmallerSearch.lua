local NormCrossMapCorrelationSmallerSearch, parent = torch.class('nn.NormCrossMapCorrelationSmallerSearch', 'nn.Module');

require 'io'
require 'cutorch'
require 'torch'
ffi = require("ffi")
ffi.cdef[[
    void NCMCSS_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, int patchwidth, int verticalWidth, 
                      THCudaTensor *meanMaps, THCudaTensor *stdMaps);
    void NCMCSS_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *gradOutput, 
                        THCudaTensor *gradInput, int patchwidth, int verticalWidth, THCudaTensor *meanMaps, THCudaTensor *stdMaps);
]]

--constructor for NormCrossMapCorrelation
function NormCrossMapCorrelationSmallerSearch:__init(patchwidth, verticalWidth)
    parent.__init(self)
    
    self.patchwidth = patchwidth
    self.verticalWidth = verticalWidth
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.gradOutput = torch.Tensor()
    self.meanMaps = torch.Tensor()
    self.stdMaps = torch.Tensor()
end

--[[
   
   name: updateOutput
   @param input - 50 layers of 37 x 12 patches
   @return - output of tensor of 50 layers of 37x12 of {(patchwidth) x {mapWidth + patchWidth - 1})}
   for example, if patchwidth = 5, then 37 x (5) x 12 x (12 + 5 - 1)
   
]]--
-- override the predefined methods
function NormCrossMapCorrelationSmallerSearch:updateOutput(input)
  local cutorchState = cutorch.getState()
  cbind = ffi.load(paths.dirname(paths.thisfile()) .. "/libNormCrossMapCorrelationSmallerSearch.so");
  cbind.NCMCSS_updateOutput(
                              cutorchState,
                              input:cdata(), 
                              self.output:cdata(),
                              self.patchwidth,
                              self.verticalWidth,
                              self.meanMaps:cdata(),
                              self.stdMaps:cdata());
  cbind = ffi.NULL;
  return self.output;
end

-- API to determine the gradient of output w.r.t., input
function NormCrossMapCorrelationSmallerSearch:updateGradInput(input, gradOutput)
  self.gradOutput:resizeAs(gradOutput):copy(gradOutput);
  local cutorchState = cutorch.getState()
  cbind = ffi.load(paths.dirname(paths.thisfile()) .. "/libNormCrossMapCorrelationSmallerSearch.so")
  cbind.NCMCSS_updateGradInput(
                              cutorchState,
                              input:cdata(),
                              self.output:cdata(),
                              self.gradOutput:cdata(),
                              self.gradInput:cdata(), 
                              self.patchwidth, 
                              self.verticalWidth,
                              self.meanMaps:cdata(),
                              self.stdMaps:cdata());
  cbind = ffi.NULL;
  return self.gradInput;
end
