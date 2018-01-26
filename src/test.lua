--[[
   test.lua
   
   Copyright 2015 Arulkumar <arul.csecit@ymail.com>
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301, USA.
   
   
]]--
----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nngraph'
require 'nn'
require 'utilities'
require 'image'
logger = require 'log'
matio = require 'matio'
dofile 'rankScores.lua'

logger.outfile = opt.logFile

----------------------------------------------------------------------
logger.trace('==> defining test procedure')


-- test function
function test(isDisplay)
    -- define if logging trace display is required
    if(isDisplay == nil) then isDisplay = true; end
 
    --do return end
    -- local vars
    local time = sys.clock()
    
    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model = model:cuda()
    model:evaluate()
    
    outputNodes = 2
    if(opt.loss == 'margin') then outputNodes = 1 end 
    
    outputs = torch.DoubleTensor(#testPairs, outputNodes):zero()
    targets = torch.DoubleTensor(#testPairs):zero()
  
    local f = 0;
    
    --forward all the test pairs and print the confusion matrix
    for index = 1, #testPairs do
      -- disp progress
      xlua.progress(index,  #testPairs)     
      currentPair = testPairs[index]['pair']
      currentPair = cudaTheData(currentPair)
      target = testPairs[index]['target']
      
      if(opt.loss == 'margin') then 
        target = torch.CudaTensor({target});
        target[target:eq(2)] = -1;
      end
	  
      local output = localizeMemory(model:forward(currentPair))
      local err = criterion:forward(output, target)
      f = f + err;
      currentScore = output
      
      if(opt.scoreregularizer == true and opt.loss == 'nll') then
        -- add (log p)2 to the error
        --view the target as a Tensor
        targetNow = target 
        outputNow = output

        if(opt.nGPUs == 1 and opt.loss == 'nll') then 
          outputNow = outputNow:view(1, 2)
          targetNow = torch.Tensor({target}) 
        end
        
        --gather all the scores depending on the type of examples +ves, -ves
        indices = targetNow:typeAs(outputNow):view(targetNow:size(1), 1)
        classSpecificScores = outputNow:gather(2, indices)
        
        --add the gradient to current error
        f = f + torch.sum(torch.pow(classSpecificScores, 2)) / 2
      end
      
	  -- add the gradient for variance reduction if needed
	  if(opt.separationCriterion == true) then
		local currentOutput = output
		local currentTarget = target
		
		-- workaround with view, incase the number of GPUs used is 1
		if(opt.nGPUs == 1) then 
		  currentOutput = output:view(1, 2) 
		  currentTarget = torch.Tensor({target}):typeAs(currentOutput)
		end
		
		local scoreDifference = currentOutput[{{},{1}}] - currentOutput[{{},{2}}]
		local Y = currentTarget:clone();
		Y[Y:eq(2)] = -1;
		
		hingeError = hingeCriterion:forward(scoreDifference, Y);
		f = f + hingeError
	  end      
      
      if(opt.loss == 'nll') then 
        confusion:add(output, testPairs[index]['target'])
      end
	  
      outputs[index][1] = currentScore[1]
      --print(currentScore)
      --print(currentScore[1])
      -- print(outputs[index][1])
      --io.read()
      if(outputNodes == 2) then outputs[index][2] = currentScore[2] end
      targets[index] = testPairs[index]['target']
    end
          
    -- print confusion matrix
    if(opt.loss == 'nll') then logger.info(confusion) end
    logger.info("error = " .. f)
    
    -- next iteration:
    confusion:zero()
    
    if(opt.testmode == 'test') then
      matio.save(paths.concat(opt.save, opt.traindataset .. '-' .. opt.testdataset .. '-scores.mat'), {output=outputs, target=targets})
    end
    
    return outputs, targets;
end
