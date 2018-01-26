-- script to test webcam release dataset using the models from deepcompare, PNNet
-- normxcorr+cin, GLoss model

package.path = "../src/modules/?.lua;" .. package.path
package.path = "../src/?.lua;" .. package.path
package.cpath = "../src/modules/?.so;" .. package.cpath
require 'torch'
require 'optim'
require 'io'
require 'cutorch'
require 'nngraph'
require 'cunn'
require 'lfs'
require 'xlua'
require 'gnuplot'
logger = require 'log'
require 'utilities'
require 'NormCrossMapCorrelation'
require 'NormCrossMapCorrelationAcrossMaps'
require 'CrossInputNeighborhood'
require 'NormCrossMapCorrelationSmallerSearch'
matio = require 'matio'
local Threads = require 'threads'
dofile 'allmodels.lua'

opt = {}
opt.scale = {1}

local SAME = 1
local DIFFERENT = 2

trainSet = 'notredame';
MODEL_TYPE = 'deepcompare_illumaug_2ch2stream' -- normxcorr_cin | deepcompare | ---
MODEL_PATH = MODELS[trainSet .. '_' .. MODEL_TYPE]

-- load the model
local model = torch.load(MODEL_PATH):cuda()

datasetName = 'webcamRelease_pre15'
datasetPath = '../datasets/' .. datasetName
cacheFile = datasetName .. '-cache.t7'

local sortFunction = function (a, b)
						return string.lower(a) < string.lower(b)
					 end

--This has to be done only once and global for all datasets
if(path.exists(cacheFile) == false) then
	sceneNames, sceneFolderPaths = getAllFileNamesInDir(datasetPath, true);
	imageBuffer = {}

	for index, folderPath in ipairs(sceneFolderPaths) do
		imageBuffer[sceneNames[index]] = {}
		
		--for each folder, get the day folder names
		dayNames, dayFolderPaths = getAllFileNamesInDir(folderPath, true);
		table.sort(dayNames, sortFunction)
		imageBuffer[sceneNames[index]]['daynames'] = dayNames
		
		--for each day, get all the image names and store in the hash
		kpNames, _ = getAllFileNamesInDir(dayFolderPaths[1], false);
		table.sort(kpNames, sortFunction)
		imageBuffer[sceneNames[index]]['kpnames'] = kpNames
	end	
	
	imagePairs = {}
	targets = {}
	
	--create positive and negative pairs
	for imageType, contents in pairs(imageBuffer) do

		local days = contents['daynames']
		local kpNames = contents['kpnames']
		local referenceDay = days[1]
		
		--for each of the other days than the reference, compare the images
		for kpIndex, kpName in ipairs(kpNames) do
			for dayIndex = 2, #days do
				-- for each day, insert a positive pair and a negative pair
				img1path = paths.concat(datasetPath, imageType, referenceDay, kpName)
				img2path = paths.concat(datasetPath, imageType, days[dayIndex], kpName)
				
				table.insert(imagePairs, {img1path, img2path});
				table.insert(targets, 1)
				
				negativeKpIndex = getRandomNumber(1, #contents['kpnames'], kpIndex)
				negativeKpDay = getRandomNumber(1, #contents['daynames'], 1)
				negImgPath = paths.concat(datasetPath, imageType, days[negativeKpDay], kpNames[negativeKpIndex])
	
				table.insert(imagePairs, {img1path, negImgPath});
				table.insert(targets, 2)
			end
		end
	end
	
	config = {}; 
	config['images'] = imagePairs
	config['targets'] = targets
	torch.save(cacheFile, config)
else
	logger.trace('cache file ' .. cacheFile .. ' found! reading the data')
	config = torch.load(cacheFile)
	imagePairs = config['images']
	targets = config['targets']
end

logger.trace('Number of image pairs : ' .. #imagePairs);

--[[
for index = 1, #imagePairs do
	print(imagePairs[index])
	print(targets[index])
	io.read()
end
--]]

--for each image pair, calculate the score and write it into the buffer file
outputTensor = nil
targetTensor = torch.zeros(#targets)

if(string.match(MODEL_TYPE, 'deepcompare')) then
	outputTensor = torch.zeros(#imagePairs, 1)
else
	outputTensor = torch.zeros(#imagePairs, 2)
end
	
logger.trace('testing for model : ' .. MODEL_PATH)

for index, imgPair in ipairs(imagePairs) do
	xlua.progress(index, #imagePairs)
	img1 = image.load(string.gsub(imgPair[1], '/data/arul', '/data/arul/arbeiten')) -- 1x65x65
	img2 = image.load(string.gsub(imgPair[2], '/data/arul', '/data/arul/arbeiten')) -- 1x65x65
	img1 = img1[{{1},{1,64},{1,64}}] -- 1x64x64
	img2 = img2[{{1},{1,64},{1,64}}] -- 1x64x64
	
	targetTensor[index] = targets[index]
	
	-- record the score (probability, direct score, euclidian distance ...)
	if(string.match(MODEL_TYPE, 'deepcompare')) then
		local inputData = torch.zeros(1, 2, 64, 64)
		
		--assign input data to 4D buffer
		inputData[1][1] = img1 - torch.mean(img1);
		inputData[1][2] = img2 - torch.mean(img2);
		
		local score = model:forward(inputData:cuda()) 
		outputTensor[index][1] = score[1][1]
		--print(score)
		--io.read()
	else
		local pred = model:forward({img1:cuda(), img2:cuda()})
		outputTensor[index][1] = pred[1]
		outputTensor[index][2] = pred[2]
	end
end

--after the test has been done, rearrange the tables into a tensor
--save the scores and targets
matio.save(MODEL_TYPE .. '-train-' .. trainSet .. '-test-' .. datasetName .. '-scores.mat', {output=outputTensor, target=targetTensor})
--]]
