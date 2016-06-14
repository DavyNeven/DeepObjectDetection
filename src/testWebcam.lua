require 'cudnn'
require 'image'
require 'gnuplot'

local MultiScaleDetector = require 'MultiScaleDetector'

local cv = require 'cv'
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video stream
require 'cv.imgproc' -- Image processing (resize, crop, draw text, ...)
require 'nn'

local capture = cv.VideoCapture{device=2}
if not capture:isOpened() then
   print("Failed to open the default camera")
   os.exit(-1)
end

-- Create a new window
cv.namedWindow{winname="Torch-OpenCV Traffic sign detection demo", flags=cv.WINDOW_AUTOSIZE}
-- Read the first frame
local _, frame = capture:read{}

-- Load model
local model = torch.load('MultiTaskModel.t7')
model:cuda()
model:evaluate()

MultiScaleDetector.init(model)

while true do
   local w = frame:size(2)
   local h = frame:size(1)
   local im = frame:clone()
   -- Get central square crop
   --local crop = cv.getRectSubPix{frame, patchSize={h,h}, center={w/2, h/2}}
   local I = im:float():div(255)
   cv.cvtColor{I, dst=I, code=cv.COLOR_BGR2RGB}
   -- Resize it to 256 x 256
   local I = I:permute(3,1,2)

   
   local BBs = MultiScaleDetector.doMultiScaleDetection(I)
   print(BBs)
   if(BBs:dim() ~= 0) then
     for i=1,BBs:size(1) do
        local rect = BBs[i]
        cv.rectangle{im, {rect[1], rect[2]}, {rect[3], rect[4]}, color={0,0,255,0}}
     end
   end
   
   -- Show it to the user
   cv.imshow{"Torch-OpenCV Traffic sign detection demo", im}
   if cv.waitKey{10} >= 0 then break end

   -- Grab the next frame
   capture:read{frame}
end