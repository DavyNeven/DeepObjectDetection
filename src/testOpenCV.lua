require 'cudnn'
require 'image'

local MultiScaleDetector = require 'MultiScaleDetector'

local cv = require 'cv'
require 'cv.highgui' -- GUI
require 'cv.imgproc' -- Image processing (resize, crop, draw text, ...)
require 'cv.imgcodecs'

-- Create a new window
cv.namedWindow{winname="Torch-OpenCV Traffic sign detection demo", flags=cv.WINDOW_AUTOSIZE}

-- Load model
local model = torch.load('MultiTaskModel.t7')
model:cuda()
model:evaluate()

MultiScaleDetector.init(model)

local im  = cv.imread{'00001.ppm'}
local I = im:clone()
I = I:float():div(255)
cv.cvtColor{I, dst=I, code=cv.COLOR_BGR2RGB}
I = I:permute(3,1,2)

local BBs = MultiScaleDetector.doMultiScaleDetection(I)
 print(BBs)
 if(BBs:dim() ~= 0) then
   for i=1,BBs:size(1) do
      local rect = BBs[i]
      cv.rectangle{im, {rect[1], rect[2]}, {rect[3], rect[4]}, color={255,0,255,0}}
   end
 end
 
 -- Show it to the user
 cv.imshow{"Torch-OpenCV Traffic sign detection demo", im}


