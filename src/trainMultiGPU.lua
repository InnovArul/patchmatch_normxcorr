--[[
   train.lua
   
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

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'io'

require 'utilities'   -- user defined helper methods
logger = require 'log'

-------------------------------------------------------------------------
logger.trace '==> defining some tools'

local SAME = 1
local DIFFERENT = 2

-- classes
classes = {'same', 'different'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainLogger.showPlot = false; trainLogger.epsfile = paths.concat(opt.save, 'train.eps')

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger.showPlot = false; testLogger.epsfile = paths.concat(opt.save, 'train.eps')

----------------------------------------------------------------------
logger.trace '==> configuring optimizer'

if opt.optimization == 'CG' then
    optimState = {
        maxIter = opt.maxIter
    }
    optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
    optimState = {
        learningRate = opt.learningRate,
        maxIter = opt.maxIter,
        nCorrection = 10
    }
    optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay
    }
    optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
    optimState = {
        eta0 = opt.learningRate,
        t0 = trsize * opt.t0
    }
    optimMethod = optim.asgd

elseif opt.optimization == 'ADAM' then
    optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        learningRateDecay = opt.learningRateDecay
    }
    optimMethod = optim.adam

else
    error('unknown optimization method')
end

----------------------------------------------------------------------

logger.trace '==> defining training procedure'

function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    --make DataParallelTable 
    model = makeDataParallel(model, opt.nGPUs) 
    
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- get the handles for parameters and gradient parameters
    parameters,gradParameters = model:getParameters()

    -- do one epoch
    logger.trace('==> doing epoch on training data:')
    logger.trace("==> online epoch # " .. epoch)

    -- create input pair combinations
    local inputs = {}
    local targets = {}
    numPositives = 0;
    numNegatives = 0;  
    totalFiles = 0;
    prevEpochNegExamples = 0;
        
    --insert the negative samples into input data if they have been collected in previous epoch
    if(negativeExamples ~= nil and #negativeExamples > 0) then
      logger.trace(#negativeExamples .. ' have been found in previous epoch as unsuccessful predictions, now prepending to the current epoch input')
      prevEpochNegExamples = #negativeExamples
      
      for i = 1, #negativeExamples do
        table.insert(inputs, negativeExamples[i]['pair'])
        table.insert(targets, negativeExamples[i]['target'])		
      end
    end
    
    -- separate the inputs and targets
    for i = 1, #trainPairs do
        table.insert(inputs, trainPairs[i]['pair'])
        table.insert(targets, trainPairs[i]['target'])
    end      
    
     ---- after generating all positive, negative pairs, split them into batch size of opt.batchSize
    -- then for each batch size of opt.batchSize, do the stochastic gradient descent
    totalSamples = #targets
    totalBatches = totalSamples / opt.batchSize;

    -- if the totalSamples count is not divisble by opt.batchSize, then add +1
    if(totalSamples % opt.batchSize ~= 0) then
        totalBatches = math.floor(totalBatches + 1)
    end
        
    logger.debug('total pairs of training samples : ' .. totalSamples .. ' (total : ' .. totalFiles .. ' / positives: ' .. numPositives .. ' / negatives: ' .. numNegatives .. '), total batches: ' .. totalBatches)
    --io.read()

    -- randomize the generated inputs and outputs
    randomOrder = torch.randperm(totalSamples)
    --modify the random order if negative exampls have been found in last epoch
    if(negativeExamples ~= nil and #negativeExamples > 0) then
		logger.trace('redefining randomOrder as per unsuccessful examples found in last epoch')
		negExamplesPerm = torch.randperm(#negativeExamples)
		originalExamplesPerm = torch.randperm(#trainPairs) + #negativeExamples
		
		randomOrder = torch.zeros(#trainPairs + #negativeExamples)
		randomOrder[{{1, #negativeExamples}}] = negExamplesPerm
		randomOrder[{{#negativeExamples + 1, #trainPairs + #negativeExamples}}] = originalExamplesPerm
    end    

    --restrict batch count if requested
    if(opt.restrictedBatchCount and totalBatches > opt.restrictedBatchCount) then
        logger.trace ('\n\nrestricting batch count to ' .. opt.restrictedBatchCount .. '\n')
        totalBatches = opt.restrictedBatchCount;
    end
    
    local currentEpochError = 0;
    local currentIntervalError = 0;
    local printErrorInterval = 500;
    local sigmaPlus = 0
    local muPlus = 0
    local sigmaMinus = 0
    local muMinus = 0;
    
    --empty the negative examples list, so that new examples can be collected in this epoch
    negativeExamples = {} 
    
    if(opt.globalloss == true) then logger.trace('using global loss'); 	end
    if(opt.separationCriterion == true) then logger.trace('using score separation criterion with margin ' .. opt.separationmargin); 	end
	if(opt.scoreregularizer == true) then logger.trace('using score regularized to reduce the variance of scores'); 	end
	if(opt.negativemining == true) then logger.trace('Negative mining is done at each epoch where unsuccessful examples from previous epoch will be prepended for current training examples'); 	end
        
    totalSamplesSoFar = 0
    logger.trace(optimState)    

     ---- for each batch, do the SGD
    for batchIndex = 0, totalBatches - 1 do
        -- disp progress
        xlua.progress(batchIndex + 1, totalBatches)
        
        -- find the batchsamples start index and end index
        time = sys.clock()
        local batchStart = (batchIndex  * opt.batchSize) + 1
        local batchEnd = ((batchIndex + 1)  * opt.batchSize);

        -- make sure that index do not exceed the end index of the totalSamples
        if(batchEnd > totalSamples) then
            batchEnd = totalSamples
        end

        local currentBatchSize = batchEnd - batchStart + 1;

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            model:zeroGradParameters()
            
            --collect all the training data and targets
            local currentInputs = {}
            local currentTargets = {}
            for dataIndex = batchStart, batchEnd do
              local currentData = inputs[randomOrder[dataIndex]]
              local currentY = targets[randomOrder[dataIndex]]
              local transformedImage = currentData
              table.insert(currentInputs, transformedImage); table.insert(currentTargets, currentY);
            end
            
            --augment the data
            currentInputs = batchRandomTransform(currentInputs)
            
            --copy the transformed data to actual place so that the data is used in normal forward backward prop
            for dataIndex = batchStart, batchEnd do
              trainPairs[randomOrder[dataIndex]]['pair'] = currentInputs[dataIndex - batchStart + 1]
            end
            
            -- f is the average of all criterions
            local f = 0
            currentTotalImagesTrained = 0;
            --arrangedInputs = getArrangedInputsForNGPUs(inputs, targets, opt.nGPUs, batchStart, batchEnd, randomOrder)
            arrangedInputs = getArrangedInputsForNGPUs(currentInputs, currentTargets, opt.nGPUs, 
                                                      1, #currentTargets, torch.randperm(#currentTargets))

            
            -- evaluate function for complete mini batch
            for i, trainset in ipairs(arrangedInputs) do

              input = trainset.input
              target = trainset.labels
              currentNumOfImages = target:size(1)
              if(opt.nGPUs == 1) then 
                input = input[1]; 
                if(opt.loss == 'nll') then target = target[1] end
              end        
				
              --make the target as -1 for margin loss
              if(opt.loss == 'margin') then
                target[target:eq(2)] = -1;
              end
                
                    -- estimate f
              local output = localizeMemory(model:forward(input))
              local err = criterion:forward(output, target)
              
              --collect unsuccessful examples
              if(opt.negativemining == true and opt.loss == 'nll' and totalSamplesSoFar >= prevEpochNegExamples) then
				targetNow = target
				outputNow = output
				inputNow = input
				
				if(opt.nGPUs == 1) then 
					outputNow = outputNow:view(1, 2)
					targetNow = torch.Tensor({target}) 
					inputNow = {}
					inputNow[1] = input
				end
				
				for exampleIndex = 1, targetNow:size(1) do
					if((targetNow[exampleIndex] == 1 and outputNow[exampleIndex][1] < outputNow[exampleIndex][2]) or
						(targetNow[exampleIndex] == 2 and outputNow[exampleIndex][1] > outputNow[exampleIndex][2])) then
						currNegInput = {inputNow[exampleIndex][1]:double(), inputNow[exampleIndex][2]:double()}
						table.insert(negativeExamples, {pair=currNegInput, target=targetNow[exampleIndex]})
					end
				end
              end
              
              f = f + err
              -- estimate df/dW
              local df_do = localizeMemory(criterion:backward(output, target))
                
              if(opt.scoreregularizer == true and opt.loss == 'nll') then
                -- add (log p)2 to the error
                --view the target as a Tensor
                targetNow = target 
                outputNow = output
                gradientNow = df_do
                scoreRegGradient = gradientNow.new():resizeAs(gradientNow):zero()
                if(opt.nGPUs == 1 and opt.loss == 'nll') then 
                  outputNow = outputNow:view(1, 2)
                  targetNow = torch.Tensor({target}) 
                  gradientNow = df_do:view(1,2)
                  scoreRegGradient = scoreRegGradient:view(1,2)
                end
                
                --gather all the scores depending on the type of examples +ves, -ves
                indices = targetNow:typeAs(outputNow):view(targetNow:size(1), 1)
                classSpecificScores = outputNow:gather(2, indices)
                
                --add the gradient to current error
                f = f + torch.sum(torch.pow(classSpecificScores, 2)) / 2
                
                --score regularizer gradient
                scoreRegGradient = scoreRegGradient:scatter(2, indices, classSpecificScores)
                
                --add the gradient to already existing gradient
                gradientNow:add(scoreRegGradient)
              end
                
              -- add the gradient for variance reduction if needed
              if(opt.separationCriterion == true) then
                local currentOutput = output
                local currentGradient = df_do
                local currentTarget = target
                
                -- workaround with view, incase the number of GPUs used is 1
                if(opt.nGPUs == 1) then 
                  currentOutput = output:view(1, 2) 
                  currentGradient = df_do:view(1, 2)
                  currentTarget = torch.Tensor({target}):typeAs(currentOutput)
                end
                
                local scoreDifference = currentOutput[{{},{1}}] - currentOutput[{{},{2}}]
                local Y = currentTarget:clone();
                Y[Y:eq(2)] = -1;
                
                hingeError = hingeCriterion:forward(scoreDifference, Y);
                f = f + hingeError
                hingeGradient = hingeCriterion:backward(scoreDifference, Y);
                
                --for same scores, add scoreDifference / N
                currentGradient[{{},{1}}] = currentGradient[{{},{1}}] + hingeGradient
                currentGradient[{{},{2}}] = currentGradient[{{},{2}}] - hingeGradient
              end
              
              model:backward(input, df_do)
              
              currentTotalImagesTrained = currentTotalImagesTrained + currentNumOfImages
			  totalSamplesSoFar = totalSamplesSoFar + currentNumOfImages;              

              -- update confusion
              if(opt.loss == 'nll') then
                if(opt.nGPUs > 1) then
                  confusion:batchAdd(output, target)
                else
                  confusion:add(output, target)
                end
              end

            end --for i = batchStart, batchEnd do

            -- normalize gradients and f(X)
            --print('total images : ' .. currentTotalImagesTrained)
            gradParameters:div(currentTotalImagesTrained)
            f = f/currentTotalImagesTrained
            
            -- add the current error to the variables
            currentIntervalError = currentIntervalError + f;
            currentEpochError = currentEpochError + f;
            
            -- return f and df/dX
            return f,gradParameters
        end
        
        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        
        -- DataParallelTable's syncParameters
       if model.needsSync then
          model:syncParameters()
       end        
       
        if(batchIndex % printErrorInterval == 0) then
            --print confusion matrix
            if(opt.loss == 'nll') then logger.trace(confusion) end
            logger.trace('error for current epoch : ' .. currentEpochError);
            logger.trace('error for current interval (' .. printErrorInterval .. ' batches): ' .. currentIntervalError);
            logger.trace('unsuccessful samples for current interval (' .. printErrorInterval .. ' batches): ' .. #negativeExamples);
            --reset error for current interval
            currentIntervalError = 0;
            
			--if(batchIndex >= 500) then break end
        end
    end -- for batchIndex = 0, totalBatches - 1 do


    -- time taken
    time = sys.clock() - time
    time = time / totalBatches
    logger.trace("\n\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- update logger/plot
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    if opt.plot then
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        --trainLogger:plot()
    end

    -- logger.trace confusion matrix
    
    if(opt.loss == 'nll') then logger.trace(confusion) end
    logger.trace('error for current epoch : ' .. currentEpochError);
    logger.trace('error for current interval (' .. printErrorInterval .. ' batches): ' .. currentIntervalError);
    logger.trace('unsuccessful samples for current interval (' .. printErrorInterval .. ' batches): ' .. #negativeExamples);
            
    -- next epoch
    confusion:zero()
    epoch = epoch + 1
      
    ----------------------------------------------------------------------
        -- save/log current net
    local filename = SAVE_MODEL_NAME .. '#' .. (epoch - 1) .. '.net';
    os.execute('mkdir -p ' .. sys.dirname(filename))
    logger.trace('==> saving model to '..filename)
    logger.trace(optimState)
    
    model = getInternalModel(model)    
    torch.save(filename, model)
end

--[[
-- logger.trace the size of the Threshold outputs
conv_nodes = model:findModules('nn.SpatialConvolutionMM')
for i = 1, #conv_nodes do
  --logger.trace(conv_nodes[i].output:size())
  --image.display(conv_nodes[i].output)
end
--]]
