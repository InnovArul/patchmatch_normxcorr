
package.path = "./modules/?.lua;" .. package.path
package.cpath = "./modules/?.so;" .. package.cpath
logger = require 'log'
dofile 'utilities.lua'

opt = {}
--torch.setdefaulttensortype('torch.CudaTensor') 
cutorch.setDevice(1)

allDatasets = {'liberty', 'yosemite', 'notredame'}
opt.useCuda = true; --true / false
opt.optimization = 'SGD'   -- CG  | LBFGS  |  SGD   | ASGD | ADAM
opt.traindatatype = 'liberty' ---- notredame | yosemite | liberty
opt.traindataset = opt.traindatatype .. '_500K'; ---- notredame_500K | yosemite_500K | liberty_500K | oxford
opt.testdataset = 'yosemite_100K'; -- -- notredame_100K | yosemite_100K | liberty_100K | oxford
opt.datapath = '../datasets/'
opt.dataType = '' -- labeled | detected
opt.testmode = 'validation' -- validation | test
opt.learningRate = 0.05 -- 0.05 -- 
opt.weightDecay = 5e-4
opt.momentum = 0.9
opt.learningRateDecay = 1e-4
opt.batchSize = 128
opt.modelType = 'cin_ss_conv5_ncc5_1maxpool' -- normxcorr_acrossmaps_scale{1,0.75,0.5} | normxcorr_acrossmaps_scale{1}' -- normxcorr | cin+normxcorr | cin+normxcorr_ss_conv5_ncc5_match+desc
         -- cin+normxcorr_assignorient | cin+normxcorr_ss_rotationinvariant | model_cin+normxcorr_ss_conv5_ncc3
         -- cin+normxcorr_ss_conv5_ncc5_cs | cin+normxcorr_ss_conv5_ncc5_gauss  |  cin+normxcorr_ss_conv5_ncc5_cs
         -- cin+normxcorr_ss_conv5_ncc5_hinge | cin+normxcorr_ss_conv5_ncc5_1maxpool
opt.xnormcorrEps = 0.01 
opt.scale = {1}
opt.traintype = ''
opt.plot = true
opt.GPU = 1
opt.nGPUs = 2
opt.description = 'ablation study'
opt.GlossLambda = 1
opt.GlossM = 2
opt.globalloss = false

--opt.scoreregularizer = true
--opt.margin = 10
--opt.separationCriterion = true
--opt.separationmargin = 1.4

opt.negativemining = false
opt.negativeminingepoch = 25
--opt.dataAugment = false
opt.resolution = 64 --for both height and width
opt.validationPercentage = 0.1
opt.loss = 'nll'

rootLogFolder = paths.concat(lfs.currentdir() .. '/../', 'scratch', opt.traindataset)

-------------setting for new model or finetuning --------------------------
opt.forceNewModel = true

if(opt.forceNewModel == true) then
	opt.save = paths.concat(rootLogFolder, os.date("%d-%b-%Y-%X-") .. 'compare_' .. opt.modelType .. '_' .. opt.traindataset .. '_' .. opt.dataType);
	LOAD_MODEL_NAME = paths.concat(opt.save, opt.modelType.. '_' .. opt.traindataset .. '_' .. opt.dataType)
	epoch = 1;
else
	--redefine opt.save and epoch number
	LOAD_MODEL_NAME = '../scratch/liberty_500K/09-Oct-2017-19:46:56-compare_normxcorr_ss_conv5_ncc5_1maxpool_liberty_500K_/normxcorr_ss_conv5_ncc5_1maxpool_liberty_500K_#25.net'
	epoch = 26;
	opt.save = getParentPath(LOAD_MODEL_NAME)
end

--opt.save = '/media/data/arul/normxcorr_compare/scratch/liberty/16-Jan-2017-20:30:46-personreid_normxcorr_acrossmaps_scale{1,0.75,0.5}_liberty_'
--if the save folder doesnot exist, create one
if(not isFolderExists(opt.save)) then
    paths.mkdir(opt.save)
end

SAVE_MODEL_NAME = paths.concat(opt.save, opt.modelType.. '_' .. opt.traindataset .. '_' .. opt.dataType)

--copy the data to opt.save	
datalinkPaths = '../datasets/links'
trainDatasetLinkFile = opt.traindataset .. '-train.config'
trainDatasetLinkPath = paths.concat(datalinkPaths, trainDatasetLinkFile)
logger.trace(trainDatasetLinkPath)
if(path.exists(trainDatasetLinkPath)) then
	logger.trace('training dataset found! copying linking data set')
	os.execute("ln -s " .. trainDatasetLinkPath .. ' ' .. opt.save)
end

--copy test data links
for i, name in ipairs(allDatasets) do 
	testDatasetLinkFile = name .. '_100K-test.config'
	testDatasetLinkPath = paths.concat(datalinkPaths, testDatasetLinkFile)
	if(name ~= opt.traindatatype and path.exists(testDatasetLinkPath)) then
		logger.trace(name .. '_100K dataset found! linking data set')
		logger.trace(testDatasetLinkPath)		
		os.execute("ln -s " .. testDatasetLinkPath .. ' ' .. opt.save)
	end
end

print("options read from opts.lua")

opt.logFile = paths.concat(opt.save, opt.modelType.. '_' .. opt.traindataset .. '_' .. opt.dataType .. '.log')
logger.outfile = opt.logFile;



