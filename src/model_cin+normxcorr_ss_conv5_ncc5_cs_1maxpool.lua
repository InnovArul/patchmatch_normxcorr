--[[
   model.lua
   
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

require 'torch';
require 'nn';
require 'io';
require 'lfs'
require 'cunn'
require 'cutorch'
dofile 'utilities.lua'
require 'nngraph';
require 'NormCrossMapCorrelationSmallerSearch'
require 'CrossInputNeighborhood'

--[
if(opt == nil) then
    opt = {}
    opt.forceNewModel = true
    opt.useCuda = true
    opt.logFile = 'modelCreation.log'
end
--]]--

logger = require 'log'
logger.outfile = opt.logFile

--cudnn.benchmark = true;
--cudnn.fastest = true;
--cudnn.verbose = true;

--define fillcolors for different layers
COLOR_CONV = 'cyan';
COLOR_MAXPOOL = 'grey';
COLOR_RELU = 'lightblue';
COLOR_SOFTMAX = 'green';
COLOR_FC = 'orange';
COLOR_CROSSMAP = 'yellow';
COLOR_AUGMENTS = 'brown';

TEXTCOLOR = 'black';
NODESTYLE = 'filled';

-- input dimensions:
local nfeats = 1
local width = 64
local height = 64

-- hidden units, filter sizes (for ConvNet only):
local nstates = {32,96,96,256}
--local nstates = {32,64,64,500}
local filtsize = {5,5,3,3}
local poolsize = 2
local padsize = 0
local patchSize = 5
local searchWidth = 5

nnpackage = nn;

function getPadSize(filtersize)
	return math.floor(filtersize/2)
end

-- read the image
--img = image.load('ahmed_model/1.png');

function create_model()
    nngraph.setDebug(true);
    ---check if an already saved model present in the current directory
    --if so, load the model and continue to train and test

    if(opt.forceNewModel == nil or opt.forceNewModel ~= false ) then
      LOAD_MODEL_NAME = 'any'
    end
    
    if(not opt.forceNewModel or lfs.attributes(LOAD_MODEL_NAME) ~= nil) then
        logger.trace('\n loading the existing model : ' .. LOAD_MODEL_NAME .. '\n')
        model = torch.load(LOAD_MODEL_NAME)
        model = localizeMemory(model);
        io.read()
    else
        logger.trace('\n creating new model : ' .. LOAD_MODEL_NAME .. '\n')
        
        input = nn.Identity()()
        
        input1 = nn.SelectTable(1)(input)
        input2 = nn.SelectTable(2)(input)
        
        -- code for central patch network
        
        centralclipper1 = nn.SpatialZeroPadding(-16, -16, -16, -16)(input1)
        --------------------------------------------------------------------------------
        -- Network for the image-1  (called as subNetwork1)
        --------------------------------------------------------------------------------
        --subNetwork1 Tied convolution maxpooling-I
        centralimg1_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1], 1, 1,
													getPadSize(filtsize[1]), getPadSize(filtsize[1]))(centralclipper1):annotate{
            name='Image[1] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralimg1_relu1 = nnpackage.ReLU()(centralimg1_conv1):annotate{
            name='Image[1] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        centralimg1_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralimg1_relu1):annotate{
            name='Image[1] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork1 Tied convolution maxpooling-II
        centralimg1_conv2 =nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2], 1, 1,
													getPadSize(filtsize[2]), getPadSize(filtsize[2]))(centralimg1_maxpool1):annotate{
            name='Image[1] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralimg1_relu2 = nnpackage.ReLU()(centralimg1_conv2):annotate{
            name='Image[1] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        -------------------------------------------------------------------
        -- Network for the image-2 (called as subNetwork2
        --------------------------------------------------------------------------------
        centralclipper2 = nn.SpatialZeroPadding(-16, -16, -16, -16)(input2)
        --subNetwork2 Tied convolution maxpooling-I
        centralimg2_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1], 1, 1,
													getPadSize(filtsize[1]), getPadSize(filtsize[1]))(centralclipper2):annotate{
            name='Image[2] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --share the weights with image1 conv1 layer
        centralimg2_conv1.data.module:share(centralimg1_conv1.data.module, 'weight', 'bias', 'gradWeight', 'gradBias');

        --ReLU
        centralimg2_relu1 = nnpackage.ReLU()(centralimg2_conv1):annotate{
            name='Image[2] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        centralimg2_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralimg2_relu1):annotate{
            name='Image[2] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork2 Tied convolution maxpooling-II
        centralimg2_conv2 = nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2], 1, 1,
													getPadSize(filtsize[2]), getPadSize(filtsize[2]))(centralimg2_maxpool1):annotate{
            name='Image[2] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralimg2_relu2 = nnpackage.ReLU()(centralimg2_conv2):annotate{
            name='Image[2] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --share the weights with image1 conv2 layer
        centralimg1_conv2.data.module:share(centralimg2_conv2.data.module, 'weight','bias','gradWeight','gradBias');

        -----------------------------------------------------------------------------------------------
        --Join the two parallel networks for cross input neighborhood differences layer processing
        -----------------------------------------------------------------------------------------------

        -- Joining layer to join the filtered features from two subNetworks
        centralXcorrJoin = nn.JoinTable(1)({centralimg1_relu2, centralimg2_relu2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --cross CorrelationUnit unit
        centralnormCrossCorrelationUnit = nn.NormCrossMapCorrelationSmallerSearch(patchSize, searchWidth)(centralXcorrJoin):annotate{
            name='Normalized Cross Map correlation unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CROSSMAP}
        };
        
        centralCCU_ReLU = nn.ReLU()(centralnormCrossCorrelationUnit):annotate{
            name='Normalized Cross Map correlation unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --nstates[3], nstates[3]
        centralspatialConfidence = nn.SpatialConvolution(5*5*nstates[2], nstates[2], 1, 1)(centralCCU_ReLU):annotate{
            name='Normalized Cross Map correlation confidence summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --ReLU
        centralCCU_summary_relu = nnpackage.ReLU()(centralspatialConfidence):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };        
                
        centralglobal_summary1 = nn.SpatialConvolution(nstates[2], nstates[3], filtsize[3], filtsize[3], 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(centralCCU_summary_relu):annotate{
            name='Normalized Cross Map correlation summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --ReLU
        centralglobal_summary1_relu = nnpackage.ReLU()(centralglobal_summary1):annotate{
            name='Normalized Cross Map correlation summary features - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };    
        
        centralglobal_summary1_maxpool = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralglobal_summary1_relu):annotate{
            name='Normalized Cross Map correlation summary maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        centralglobal_summary2 = nn.SpatialConvolution(nstates[3], nstates[3], filtsize[3], filtsize[3], 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(centralglobal_summary1_maxpool):annotate{
            name='Normalized Cross Map correlation summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --ReLU
        centralglobal_summary2_relu = nnpackage.ReLU()(centralglobal_summary2):annotate{
            name='Normalized Cross Map correlation summary features - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };    
        
        centralglobal_summary2_maxpool = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralglobal_summary2_relu):annotate{
            name='Normalized Cross Map correlation summary maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };        
        
        --reshape
        centralreshapeXcorr = nn.Reshape(nstates[3]*4*4)(centralglobal_summary2_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC
        centralFC_XCORR = nn.Linear(nstates[3]*4*4, nstates[4])(centralreshapeXcorr):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

    -------------------------------------------------------------------------------------------------
    -- Ahmed paper branch
       -- Joining layer to join the filtered features from two subNetworks
        centralCinJoin = nn.JoinTable(1)({centralimg1_relu2, centralimg2_relu2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --Cross input neighborhood differences
        centralCIN = nn.CrossInputNeighborhood()(centralCinJoin):annotate{
            name='Cross Input Neighborhood unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --join
        centraljoining = nn.JoinTable(1)(centralCIN):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --CrossInputNeighborhood ReLU
        centralCINrelu = nnpackage.ReLU()(centraljoining):annotate{
            name='Cross Input Neighborhood unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Patch summary features
        centralpatch_summary = nnpackage.SpatialConvolution(nstates[2] * 5 * 5 * 2, nstates[2], 1, 1)(centralCINrelu):annotate{
            name='Patch summary features[1] - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralpatch_summary_relu = nnpackage.ReLU()(centralpatch_summary):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Across patch features
        centralacross_patch1 = nnpackage.SpatialConvolution(nstates[2], nstates[3], 3, 3, 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(centralpatch_summary_relu):annotate{
            name='Across patch features - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralacross_patch1_relu = nnpackage.ReLU()(centralacross_patch1):annotate{
            name='Across patch features - ReLU unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };


        --Maxpool
        centralacross_patch1_maxpool = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralacross_patch1_relu):annotate{
            name='Across patch features - Maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };


        --Across patch features
        centralacross_patch2 = nnpackage.SpatialConvolution(nstates[3], nstates[3], 3, 3, 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(centralacross_patch1_maxpool):annotate{
            name='Across patch features - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        centralacross_patch2_relu = nnpackage.ReLU()(centralacross_patch2):annotate{
            name='Across patch features - ReLU unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };


        --Maxpool
        centralacross_patch2_maxpool = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(centralacross_patch2_relu):annotate{
            name='Across patch features - Maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        ----------------------------------------------------------------------
        -- join -> reshape -> FC500
        ----------------------------------------------------------------------------------------------------
	
        --reshape
        centralreshapeCin = nn.Reshape(nstates[3]*4*4)(centralacross_patch2_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC
        centralFC_CIN = nn.Linear(nstates[3]*4*4, nstates[4])(centralreshapeCin):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

        centralFC_Join = nn.JoinTable(1)({centralFC_XCORR, centralFC_CIN});
        
        --ReLU
        centralFCReLU = nnpackage.ReLU()(centralFC_Join):annotate{
            name='Fully connected layer - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

		-------------------------------------------------------------------------------------------
		-- Stream branch for Surrounding patch
		
		-- code for Surrounding patch network
        
        surroundImage1 = nn.SpatialAveragePooling(poolsize, poolsize, poolsize, poolsize)(input1)
        --------------------------------------------------------------------------------
        -- Network for the image-1  (called as subNetwork1)
        --------------------------------------------------------------------------------
        --subNetwork1 Tied convolution maxpooling-I
        surroundimg1_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1], 1, 1,
													getPadSize(filtsize[1]), getPadSize(filtsize[1]))(surroundImage1):annotate{
            name='Image[1] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundimg1_relu1 = nnpackage.ReLU()(surroundimg1_conv1):annotate{
            name='Image[1] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        surroundimg1_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundimg1_relu1):annotate{
            name='Image[1] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork1 Tied convolution maxpooling-II
        surroundimg1_conv2 =nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2], 1, 1,
													getPadSize(filtsize[2]), getPadSize(filtsize[2]))(surroundimg1_maxpool1):annotate{
            name='Image[1] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundimg1_relu2 = nnpackage.ReLU()(surroundimg1_conv2):annotate{
            name='Image[1] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        -------------------------------------------------------------------
        -- Network for the image-2 (called as subNetwork2
        --------------------------------------------------------------------------------
        surroundImage2 = nn.SpatialAveragePooling(poolsize, poolsize, poolsize, poolsize)(input2)
        
        --subNetwork2 Tied convolution maxpooling-I
        surroundimg2_conv1 = nnpackage.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1], 1, 1,
													getPadSize(filtsize[1]), getPadSize(filtsize[1]))(surroundImage2):annotate{
            name='Image[2] - Convolution unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --share the weights with image1 conv1 layer
        surroundimg2_conv1.data.module:share(surroundimg1_conv1.data.module, 'weight', 'bias', 'gradWeight', 'gradBias');

        --ReLU
        surroundimg2_relu1 = nnpackage.ReLU()(surroundimg2_conv1):annotate{
            name='Image[2] - ReLU unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --Maxpool
        surroundimg2_maxpool1 = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundimg2_relu1):annotate{
            name='Image[2] - Maxpooling unit(1)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        --subNetwork2 Tied convolution maxpooling-II
        surroundimg2_conv2 = nnpackage.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2], 1, 1,
													getPadSize(filtsize[2]), getPadSize(filtsize[2]))(surroundimg2_maxpool1):annotate{
            name='Image[2] - Convolution unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundimg2_relu2 = nnpackage.ReLU()(surroundimg2_conv2):annotate{
            name='Image[2] - ReLU unit(2)',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --share the weights with image1 conv2 layer
        surroundimg1_conv2.data.module:share(surroundimg2_conv2.data.module, 'weight','bias','gradWeight','gradBias');

        -----------------------------------------------------------------------------------------------
        --Join the two parallel networks for cross input neighborhood differences layer processing
        -----------------------------------------------------------------------------------------------

        -- Joining layer to join the filtered features from two subNetworks
        surroundXcorrJoin = nn.JoinTable(1)({surroundimg1_relu2, surroundimg2_relu2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --cross CorrelationUnit unit
        surroundnormCrossCorrelationUnit = nn.NormCrossMapCorrelationSmallerSearch(patchSize, searchWidth)(surroundXcorrJoin):annotate{
            name='Normalized Cross Map correlation unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CROSSMAP}
        };
        
        surroundCCU_ReLU = nn.ReLU()(surroundnormCrossCorrelationUnit):annotate{
            name='Normalized Cross Map correlation unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        --nstates[3], nstates[3]
        surroundspatialConfidence = nn.SpatialConvolution(5*5*nstates[2], nstates[2], 1, 1)(surroundCCU_ReLU):annotate{
            name='Normalized Cross Map correlation confidence summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --ReLU
        surroundCCU_summary_relu = nnpackage.ReLU()(surroundspatialConfidence):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };        
                
        surroundglobal_summary1 = nn.SpatialConvolution(nstates[2], nstates[3], filtsize[3], filtsize[3], 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(surroundCCU_summary_relu):annotate{
            name='Normalized Cross Map correlation summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        surroundglobal_summary1_relu = nn.ReLU()(surroundglobal_summary1):annotate{
            name='Normalized Cross Map correlation summary relu unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        surroundglobal_summary1_maxpool = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundglobal_summary1_relu):annotate{
            name='Normalized Cross Map correlation summary maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        

        surroundglobal_summary2 = nn.SpatialConvolution(nstates[3], nstates[3], filtsize[3], filtsize[3], 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(surroundglobal_summary1_maxpool):annotate{
            name='Normalized Cross Map correlation summary unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };

        surroundglobal_summary2_relu = nn.ReLU()(surroundglobal_summary2):annotate{
            name='Normalized Cross Map correlation summary relu unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        surroundglobal_summary2_maxpool = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundglobal_summary2_relu):annotate{
            name='Normalized Cross Map correlation summary maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
                
        --reshape
        surroundreshapeXcorr = nn.Reshape(nstates[3]*4*4)(surroundglobal_summary2_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC
        surroundFC_XCORR = nn.Linear(nstates[3]*4*4, nstates[4])(surroundreshapeXcorr):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

    -------------------------------------------------------------------------------------------------
    -- Ahmed paper branch
       -- Joining layer to join the filtered features from two subNetworks
        surroundCinJoin = nn.JoinTable(1)({surroundimg1_relu2, surroundimg2_relu2}):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --Cross input neighborhood differences
        surroundCIN = nn.CrossInputNeighborhood()(surroundCinJoin):annotate{
            name='Cross Input Neighborhood unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --join
        surroundjoining = nn.JoinTable(1)(surroundCIN):annotate{
            name='Joining unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };
        
        --CrossInputNeighborhood ReLU
        surroundCINrelu = nnpackage.ReLU()(surroundjoining):annotate{
            name='Cross Input Neighborhood unit - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Patch summary features
        surroundpatch_summary = nnpackage.SpatialConvolution(nstates[2] * 5 * 5 * 2, nstates[2], 1, 1)(surroundCINrelu):annotate{
            name='Patch summary features[1] - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundpatch_summary_relu = nnpackage.ReLU()(surroundpatch_summary):annotate{
            name='Patch summary features[1] - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };

        --Across patch features
        surroundacross_patch1 = nnpackage.SpatialConvolution(nstates[2], nstates[3], 3, 3, 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(surroundpatch_summary_relu):annotate{
            name='Across patch features - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundacross_patch1_relu = nnpackage.ReLU()(surroundacross_patch1):annotate{
            name='Across patch features - ReLU unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --Maxpool
        surroundacross_patch1_maxpool = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundacross_patch1_relu):annotate{
            name='Across patch features - Maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };


        --Across patch features
        surroundacross_patch2 = nnpackage.SpatialConvolution(nstates[3], nstates[3], 3, 3, 1, 1,
												getPadSize(filtsize[3]), getPadSize(filtsize[3]))(surroundacross_patch1_maxpool):annotate{
            name='Across patch features - Convolution unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_CONV}
        };

        --ReLU
        surroundacross_patch2_relu = nnpackage.ReLU()(surroundacross_patch2):annotate{
            name='Across patch features - ReLU unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        
        --Maxpool
        surroundacross_patch2_maxpool = nnpackage.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize, padsize, padsize)(surroundacross_patch2_relu):annotate{
            name='Across patch features - Maxpooling unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_MAXPOOL}
        };
        ----------------------------------------------------------------------
        -- join -> reshape -> FC500
        ----------------------------------------------------------------------------------------------------
	
        --reshape
        surroundreshapeCin = nn.Reshape(nstates[3]*4*4)(surroundacross_patch2_maxpool):annotate{
            name='Reshaping unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_AUGMENTS}
        };

        --FC
        surroundFC_CIN = nn.Linear(nstates[3]*4*4, nstates[4])(surroundreshapeCin):annotate{
            name='Fully connected layer - 500 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

        surroundFC_Join = nn.JoinTable(1)({surroundFC_XCORR, surroundFC_CIN});
        
        --ReLU
        surroundFCReLU = nnpackage.ReLU()(surroundFC_Join):annotate{
            name='Fully connected layer - ReLU',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_RELU}
        };
        
        totalFC = nn.JoinTable(1)({centralFCReLU, surroundFCReLU})

        --FC2
        FC2 = nn.Linear(4 * nstates[4], 2)(totalFC):annotate{
            name='Fully connected layer - 2 nodes',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_FC}
        };

        --Softmax
        result = nnpackage.LogSoftMax()(FC2):annotate{
            name='Softmax unit',
            graphAttributes = {color = TEXTCOLOR, style = NODESTYLE, fillcolor = COLOR_SOFTMAX}
        };

        print ('model created\n');
        
        -- packaging the network into gModule (graphical module) from nngraph
        --
        model = localizeMemory(nn.gModule({input}, {result}));
        model = localizeMemory(model);
        
        graph.dot(model.fg, 'model', 'personreid_cin+normxcorr_ss_conv5_ncc5_cs')
        
        --do return end
        --
        --CHECKING THE WEIGHT SHARING BY PRINTING THE POINTERS!
        logger.trace('central patch branch')
        logger.trace('centralimg1 conv1')
        logger.trace(model.modules[4].weight:data()) -- img1 conv1
        logger.trace('centralimg2 conv1')
        logger.trace(model.modules[11].weight:data()) -- img2 conv1
        logger.trace('centralimg1 conv2')
        logger.trace(model.modules[7].weight:data()) -- img1 conv2
        logger.trace('centralimg2 conv2')
        logger.trace(model.modules[14].weight:data()) -- img2 conv2
        
        logger.trace('surround patch branch')
        logger.trace('surroundimg1 conv1')
        logger.trace(model.modules[46].weight:data()) -- img1 conv1
        logger.trace('centralimg2 conv1')
        logger.trace(model.modules[52].weight:data()) -- img2 conv1
        logger.trace('centralimg1 conv2')
        logger.trace(model.modules[49].weight:data()) -- img1 conv2
        logger.trace('centralimg2 conv2')
        logger.trace(model.modules[55].weight:data()) -- img2 conv2        

        -- forward and backward the image in both ahmed's model and newly created model 
        -- and verify the difference in outputs upto Joining layer (before CrossInputNeighborhood and FlowCalculation layers)
        --img = localizeMemory(image.scale(img, 60, 160));
        
        -- forward for New model
        --pred = localizeMemory(model:forward({img, img})); 
 
     end -- if(path.exists(MODEL_NAME))
--]]

    -- Retrieve parameters and gradients:
    -- this extracts and flattens all the trainable parameters of the mode
    -- into a 1-dim vector
    -- for future use in train.lua

    parameters,gradParameters = model:getParameters()
    logger.trace("number of parameters: " .. parameters:size(1))
    --graph.dot(model.fg, 'model', 'personreid')

end

--create_model()
