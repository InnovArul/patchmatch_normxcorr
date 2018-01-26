--[[
   data.lua
   
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

--[[-----------------------------------------------------------

preparation of data for training and testing 

data can be different types of datasets (ethz, cuhk03, viper)

---]]-------------------------------------------------------------

require 'lfs'
require 'utilities'
require 'io'
require 'torch'
require 'pl'

logger = require 'log'
logger.outfile = opt.logFile

--data buffer to be filled
trainData = {}
testData = {}
additionalGallery = {}

local SAME = 1
local DIFFERENT = 2

if((opt.traindataset == 'liberty_500K' or opt.traindataset == 'notredame_500K' or opt.traindataset == 'yosemite_500K') and
    (opt.testdataset == 'liberty_100K' or opt.testdataset == 'notredame_100K' or opt.testdataset == 'yosemite_100K')) then
	--load cuhk03 dataset
	--get all file names
  trainDataset = opt.traindataset
  testDataset = opt.testdataset
  
  if(opt.testmode == 'validation') then
    trainfilespath = opt.datapath..trainDataset..'/';
    
    fileNames = {}
    filePaths = {}
    
    trainConfigFile = paths.concat(opt.save,trainDataset..'-train.config')
    
    if(path.exists(trainConfigFile) == false) then 
        logger.trace('previous configuration (' .. trainConfigFile .. ') for training NOT found in ' .. opt.save)
        similarPairsFolder = trainfilespath .. 'simPairs/';
        dissimilarPairsFolder = trainfilespath .. 'dissimPairs/';

        logger.trace('\tcollecting all folder names from '.. similarPairsFolder)
        --now we are getting the folder names and paths inside ../seqZ
        simFileNames, simFilePaths = getAllFileNamesInDir(similarPairsFolder, true);
        logger.trace('\tNumber of simpair folders : '.. #simFilePaths)
        
        logger.trace('\tcollecting all folder names from '.. dissimilarPairsFolder)
        dissimFileNames, dissimFilePaths = getAllFileNamesInDir(dissimilarPairsFolder, true);
        logger.trace('\tNumber of dissimpair folders : '.. #dissimFilePaths)        
        
        simTrainFiles = loadAllImagesFromFolders(simFileNames, simFilePaths)
        dissimTrainFiles = loadAllImagesFromFolders(dissimFileNames, dissimFilePaths)

        simTrainPairs, dummy = getPositiveAndNegativePairs(simTrainFiles, false)
        dissimTrainPairs, dummy = getPositiveAndNegativePairs(dissimTrainFiles, false)

        --training pairs organization
        trainPairs = {}
        for index = 1, #simTrainPairs do
            local currentPair = simTrainPairs[index]
            table.insert(trainPairs, {pair = currentPair, target = SAME })
            currentPair = dissimTrainPairs[index]
            table.insert(trainPairs, {pair = currentPair, target = DIFFERENT })
        end
        
        --]]-- save the configuration
        config = {}
        config['train'] = trainPairs;
        torch.save(trainConfigFile, config)
        --do return end
    else
        -- load the configuration
        logger.trace('previous configuration (' .. trainConfigFile .. ') for training found in ' .. opt.save)
        config = torch.load(trainConfigFile)
        trainPairs = config['train'];
    end  
    
    oldTrainPairs = trainPairs;
    
    trainPairs = {}
    validationPairs = {}
    
    local totalTrainPairs = #oldTrainPairs;      
    logger.trace("total train pairs before validation splitting : " .. totalTrainPairs)
    local randomPermutations = torch.randperm(totalTrainPairs)
    local validationCount = math.floor(totalTrainPairs * opt.validationPercentage)
    local trainingIndices = randomPermutations[{{1, totalTrainPairs - validationCount}}]
    local validationIndices = randomPermutations[{{totalTrainPairs - validationCount + 1, totalTrainPairs}}]
        
	validationSetConfigFile = paths.concat(opt.save,trainDataset..'-validationsplit.config')
	if(path.exists(validationSetConfigFile)) then
		logger.trace('validation config file found! splitting data as per ' .. validationSetConfigFile)
		validationConfig = torch.load(validationSetConfigFile)
		trainingIndices = validationConfig['trainindices']
		validationIndices = validationConfig['valindices']
	end
    
    --split into train and validation
    for index = 1, trainingIndices:size(1) do
      table.insert(trainPairs, oldTrainPairs[trainingIndices[index]])
    end
    
    --delete all the pairs which are taken as validation
    for index = 1, validationIndices:size(1) do
      table.insert(validationPairs,  oldTrainPairs[validationIndices[index]])
    end

    --assign validation pairs as test pairs
    testPairs = validationPairs
    
    logger.trace("total train pairs : " .. #trainPairs)
    logger.trace("total validation pairs : " .. #validationPairs)
    
    validationConfig = {}
	validationConfig['trainindices'] = trainingIndices
	validationConfig['valindices'] = validationIndices   
	torch.save(validationSetConfigFile, validationConfig)
  elseif(opt.testmode == 'test') then
    testfilespath = opt.datapath..testDataset..'/';
    
    fileNames = {}
    filePaths = {}
    
    testConfigFile = paths.concat(opt.save, testDataset .. '-test.config')
    
    if(path.exists(testConfigFile) == false) then 
        logger.trace('previous configuration ('.. testConfigFile .. ') for testing NOT found in ' .. opt.save)
        similarPairsFolder = testfilespath .. 'simPairs/';
        dissimilarPairsFolder = testfilespath .. 'dissimPairs/';
                
        logger.trace('\tcollecting all folder names from '.. similarPairsFolder)
        --now we are getting the folder names and paths inside ../seqZ
        simFileNames, simFilePaths = getAllFileNamesInDir(similarPairsFolder, true);
        logger.trace('\tNumber of simpair folders : '.. #simFilePaths)
        
        logger.trace('\tcollecting all folder names from '.. dissimilarPairsFolder)
        dissimFileNames, dissimFilePaths = getAllFileNamesInDir(dissimilarPairsFolder, true);
        logger.trace('\tNumber of dissimpair folders : '.. #dissimFilePaths)        
        
        simTestFiles = loadAllImagesFromFolders(simFileNames, simFilePaths)
        dissimTestFiles = loadAllImagesFromFolders(dissimFileNames, dissimFilePaths)

        simTestPairs, dummy = getPositiveAndNegativePairs(simTestFiles, false)
        dissimTestPairs, dummy = getPositiveAndNegativePairs(dissimTestFiles, false)

        --testing pairs organization
        testPairs = {}
        for index = 1, #simTestPairs do
            local currentPair = simTestPairs[index]
            table.insert(testPairs, {pair = currentPair, target = SAME })
            currentPair = dissimTestPairs[index]
            table.insert(testPairs, {pair = currentPair, target = DIFFERENT })
        end
        
        -- save the configuration
        config = {}
        config['test'] = testPairs;
        torch.save(testConfigFile, config)
        
        --do return end
    else
        -- load the configuration
        logger.trace('previous configuration ('.. testConfigFile .. ') for testing found in ' .. opt.save)
        config = torch.load(testConfigFile)
        testPairs = config['test'];
      
       -- for i = 1, #testPairs do
       --     currentPair = testPairs[i]['pair'];
            --image.display(currentPair)
            --print(i .. ' = ' .. testPairs[i]['target'])
            --io.read()
       -- end
       -- io.read()
    end     
    
  else
    logger.trace('unknown testmode!');
  end
elseif(opt.traindataset == 'oxford' and opt.testdataset == 'oxford') then
    --read the oxford dataset
    --get the dataset folder names
    trainfilespath = opt.datapath .. opt.traindataset ..'/';
    trainPairs = {}
    
    trainConfigFile = paths.concat(opt.save, opt.traindataset..'-train.config')
    
    if(path.exists(trainConfigFile) == false) then
        --get all the root folder names of different combinations
        allDatasetCombinationNames, allDatasetCombinationPaths = getAllFileNamesInDir(trainfilespath, true) 
        
        --for each of the folder, load the images into memory
        for index, folderpath in ipairs(allDatasetCombinationPaths) do
            print(folderpath)
            similarPairsFolder = paths.concat(folderpath, 'simPatches_' .. opt.resolution)
            dissimilarPairsFolder = paths.concat(folderpath, 'dissimPatches_' .. opt.resolution)
            
            simFileNames, simFilePaths = getAllFileNamesInDir(similarPairsFolder, true);
            logger.trace('\tNumber of simpair folders (' .. similarPairsFolder .. ') : '.. #simFilePaths)
            
            dissimFileNames, dissimFilePaths = getAllFileNamesInDir(dissimilarPairsFolder, true);
            logger.trace('\tNumber of dissimpair folders (' .. dissimilarPairsFolder .. ') : '.. #dissimFilePaths)  
            
            simTrainFiles = loadAllImagesFromFolders(simFileNames, simFilePaths)
            dissimTrainFiles = loadAllImagesFromFolders(dissimFileNames, dissimFilePaths)
                    
            simTrainPairs, dummy = getPositiveAndNegativePairs(simTrainFiles, false)
            dissimTrainPairs, dummy = getPositiveAndNegativePairs(dissimTrainFiles, false)
            
            for index = 1, #simTrainPairs do
                local currentPair = simTrainPairs[index]
                table.insert(trainPairs, {pair = currentPair, target = SAME })
                currentPair = dissimTrainPairs[index]
                table.insert(trainPairs, {pair = currentPair, target = DIFFERENT })
            end        
        end
        
        --save the training files configuration
        config = {}
        config['train'] = trainPairs;
        torch.save(trainConfigFile, config)        
    else
        -- load the configuration
        logger.trace('previous configuration (' .. trainConfigFile .. ') for training found in ' .. opt.save)
        config = torch.load(trainConfigFile)
        trainPairs = config['train'];
    end
    
    logger.trace('Number of train pairs : ' .. #trainPairs)
    --io.read()
else

	logger.trace('unknown datasets! train: ' .. opt.traindataset .. ', test: ' .. opt.testdataset);

end
