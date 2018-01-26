--[[
   doallTest.lua
   
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
package.path = "./modules/?.lua;" .. package.path
package.cpath = "./modules/?.so;" .. package.cpath
require 'torch'
require 'optim'
require 'io'
require 'cutorch'
require 'nngraph'
require 'cunn'
require 'lfs'
require 'gnuplot'
logger = require 'log'
dofile 'utilities.lua'
require 'NormCrossMapCorrelation'
require 'NormCrossMapCorrelationAcrossMaps'
require 'CrossInputNeighborhood'
require 'NormCrossMapCorrelationSmallerSearch'
require 'TableToTensor'
require 'RotateFeatureMaps'
--require 'stn'
require 'NormCrossMapCorrelationSmallerSearchV1_1'
require 'CrossInputNeighborhoodV1_1'
require 'GaussianWeightage'
require 'L2Distance'

-- classes
classes = {'same', 'different'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

opt = {}
torch.setdefaulttensortype('torch.FloatTensor') 

opt.useCuda = true; --true
opt.traindataset = 'notredame_500K';  ---- notredame_500K | yosemite_500K | liberty_500K
opt.testdataset = 'liberty_100K'; -- notredame_100K | yosemite_100K | liberty_100K
opt.datapath = '../datasets/'
opt.testmode = 'test' -- test
opt.modelType = 'normxcorr_ss_conv5_ncc5'-- normxcorr | cin+normxcorr , normxcorr_ss_conv5_ncc5_1maxpool
opt.scale = {1}
opt.loss = 'nll' --nll | hinge
--opt.scoreregularizer = false
opt.nGPUs = 1
--opt.separationCriterion = false
--opt.separationmargin = 1.4

assert(opt.traindataset == 'oxford' or opt.traindataset ~= opt.testdataset, "train, test datasets cannot be same except oxford!")
assert((opt.traindataset ~= nil), "train dataset cannot be empty!")
assert((opt.testdataset ~= nil), "test dataset cannot be empty!")

--liberty
--MODEL_PATH = '../scratch/liberty_500K/18-Mar-2017-19:52:13-compare_cin+normxcorr_ss_conv5_ncc5_liberty_500K_/cin+normxcorr_ss_conv5_ncc5_liberty_500K_#12.net'
--notredame
MODEL_PATH = '../scratch/notredame_500K/04-Apr-2017-10:33:46-compare_cin+normxcorr_ss_conv5_ncc5_notredame_500K_/cin+normxcorr_ss_conv5_ncc5_notredame_500K_#100.net'
--yosemite
--MODEL_PATH = '../scratch/notredame_500K/10-May-2017-11:34:15-compare_cin+normxcorr_ss_conv5_ncc5_cs_notredame_500K_/cin+normxcorr_ss_conv5_ncc5_cs_notredame_500K_#68.net'
--MODEL_PATH = '../scratch/notredame_500K/16-Oct-2017-21:15:07-compare_normxcorr_ss_conv5_ncc5_1maxpool_notredame_500K_/normxcorr_ss_conv5_ncc5_1maxpool_notredame_500K_#28.net'
--'../scratch/yosemite_500K/25-Jul-2017-12:44:04-compare_normxcorr_ss_conv5_ncc5_1maxpool_yosemite_500K_/normxcorr_ss_conv5_ncc5_1maxpool_yosemite_500K_#54.net'

--'/media/data/arul/normxcorr_compare/scratch/yosemite_500K/07-May-2017-04:35:02-compare_cin+normxcorr_ss_conv5_ncc5_cs_yosemite_500K_/cin+normxcorr_ss_conv5_ncc5_cs_yosemite_500K_#50.net'

--/media/data/arul/normxcorr_compare/scratch/yosemite_500K/07-May-2017-04:28:30-compare_cin+normxcorr_ss_conv5_ncc5_cs_1maxpool_yosemite_500K_/cin+normxcorr_ss_conv5_ncc5_cs_1maxpool_yosemite_500K_#61.net
--------------------------------------------------------------------------------------------------

opt.save = paths.dirname(MODEL_PATH)
MODEL_NAME = paths.basename(MODEL_PATH, paths.extname(MODEL_PATH))
opt.logFile = paths.concat(opt.save, MODEL_NAME .. '_forCMC.log')
opt.testErrorFile = paths.concat(opt.save, MODEL_NAME .. '_forCMC.eps')
logger.outfile = opt.logFile;

logger.trace("model name : " .. MODEL_NAME)
logger.trace("log save path : " .. opt.save)
logger.trace("log file : " .. opt.logFile)
for k,v in pairs(opt) do logger.trace(k,v) end

print("press <ENTER>, if the details are correct!");
io.read()

--load the appropriate data files
dofile 'data.lua';
dofile 'loss.lua'

--add training and test module based on type of the model, if needed
local specialModelName = opt.modelType:match('[^_]-$')
logger.trace('special Model Name : ')
logger.trace(specialModelName)

if(specialModelName ~= 'match+desc'
	and specialModelName ~= 'hinge') then 
  dofile 'test.lua';
else
  dofile('test_' .. specialModelName .. '.lua');
end

local errorHistory = nil
local epochHistory = nil

logger.trace('loading model : ' .. MODEL_PATH);
model = torch.load(MODEL_PATH)
--model:cuda()
parameters,gradParameters = model:getParameters()   

avgCMC = nil
local avgError = 0

-- start testing
outputs, targets = test();
