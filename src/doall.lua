--[[
   doall.lua
   
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
require 'torch'
require 'cutorch'
require 'lfs'
logger = require 'log'
dofile 'opts.lua';
dofile 'loss.lua';
dofile 'utilities.lua'
require 'lfs'

    
assert(opt.traindataset == 'oxford' or opt.traindataset ~= opt.testdataset, "train, test datasets cannot be same except oxford!")
assert((opt.traindataset ~= nil), "train dataset cannot be empty!")
assert((opt.testdataset ~= nil), "test dataset cannot be empty!")

logger.trace('PATH DETAILS:');
logger.trace(opt.save)
logger.trace(opt.logFile)
logger.trace(LOAD_MODEL_NAME)
logger.trace(SAVE_MODEL_NAME)
logger.trace('OPTIONS:');
for k,v in pairs(opt) do logger.trace(k,v) end

print("press <ENTER>, if the details are correct!");
io.read()

dofile 'data.lua';
--do return end

--io.read()
if(opt.modelType == 'cin+normxcorr_ss_conv5_ncc5'
    or opt.modelType == 'cin+normxcorr_ss_conv5_ncc5_cs'
    or opt.modelType == 'cin+normxcorr_ss_conv5_ncc5_1maxpool'
    or opt.modelType == 'cin_ss_conv5_ncc5_1maxpool'
    or opt.modelType == 'normxcorr_ss_conv5_ncc5_1maxpool'
    or opt.modelType == 'cin+normxcorr_ss_conv5_ncc5_cs_1maxpool') then
    dofile ('model_' .. opt.modelType .. '.lua');
    
    --add training and test module based on type of the model, if needed
    local specialModelName = opt.modelType:match('[^_]-$')
    logger.trace('special Model Name : ')
    logger.trace(specialModelName)
    
    dofile 'test.lua';
    dofile 'trainMultiGPU.lua';
else 
    print('unknown model type ' .. opt.modelType .. '\n\n');
    do return end;
end

--create or load the model
create_model();

-- start training and testing
while (epoch <= 100) do
    train();
    
    --do return end
    test();
end

