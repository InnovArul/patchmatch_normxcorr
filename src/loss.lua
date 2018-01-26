--[[
   loss.lua
   
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
require 'nn'      -- provides all sorts of loss functions
dofile 'utilities.lua'
-----------------------------------------------------------------------
print '==> define loss'


if(opt.loss == 'nll') then
	-- 2-class problem
	noutputs = 2

	---------------------------------------------------------------------
	-- unaveraged criteria is used, especially for multi GPU training
	-- averaging is done at the end of the full batch of size 'opt.batchSize'
	criterion = localizeMemory(nn.ClassNLLCriterion(nil, false))
else
	-- hinge loss for difference score separation
	criterion = localizeMemory(nn.MarginCriterion())
	criterion.sizeAverage = false;
end

if(opt.separationCriterion == true) then
	hingeCriterion = localizeMemory(nn.MarginCriterion(opt.separationmargin))
	hingeCriterion.sizeAverage = false;
end	

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
