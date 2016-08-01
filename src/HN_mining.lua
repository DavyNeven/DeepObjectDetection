local pl = (require 'pl.import_into')()

require 'nms'
require 'cunn'

local HN_mining = {}

local loadBackgroundTrainingImages
local getBackgroundTrainingImages
local mapToBB
local extractPatches

HN_mining.parent_path = {--"/usr/data/dneven/MOOSlogs/moosLogs/fridayRecording/fridayRecording/log_4_12_2015_____11_28_07/"}--,
                          "/usr/data/dneven/datasets/GTSDB/TrainIJCNN2013"}
HN_mining.bg_bin = "HN_dataset.bin"

function HN_mining.get_hard_negatives(model, winSize, stride)
  local MultiScaleDetector = require 'MultiScaleDetector'
  MultiScaleDetector.init(model, winSize, stride)
  local bg_images = getBackgroundTrainingImages()
  local totalPatches = {}
  for i=1, #bg_images do
    --local output = model:forward(input:cuda())
    --output = output[1]
    local BB = MultiScaleDetector.doMultiScaleDetectionNoRegression(bg_images[i])--mapToBB(output, winSize, stride)
    if(BB:nDimension() ~= 0) then 
      --local selection = nms(BB,0.3,5)
      --BB = BB:index(1, selection)
      print("Image : " .. i .. " extracted patches: " .. BB:size(1))
      local patches = extractPatches(bg_images[i], BB, winSize)
      table.insert(totalPatches,patches)
    end
  end
  m = nn.JoinTable(1);
  local output = m:forward(totalPatches)
  return output  
end

extractPatches = function(im, BB, winSize)
  local patches = torch.Tensor(BB:size(1), 3, winSize, winSize)
  for i = 1, BB:size(1) do
      local x1 = BB[i][1]
      local y1 = BB[i][2]
      local x2 = BB[i][3]
      local y2 = BB[i][4]
      local patch = image.scale(im[{{},{y1, y2},{x1, x2}}], winSize, winSize)
      patches[i]:copy(patch)
  end
  return patches
end

getBackgroundTrainingImages = function()
  local bg_images
  if not pl.path.isfile(HN_mining.bg_bin) then
    print('Generating bin of the HN dataset')
    bg_images = loadBackgroundTrainingImages(HN_mining.parent_path)
    torch.save(HN_mining.bg_bin, bg_images)
  else
    bg_images = torch.load(HN_mining.bg_bin)
  end
  return bg_images
end

loadBackgroundTrainingImages = function(parent_path)
 print("loading negative training images")
 local images = {}
 local csv_file_name = 'bg.csv'
 local count = 1
 for j = 1, #parent_path do
   local csv_file_path = paths.concat(parent_path[j], csv_file_name)
   print(csv_file_path)
   local csv_content = pl.data.read(csv_file_path)
   for i = 1, #csv_content do
      local file = csv_content[i][1]
      local image_data = image.load(paths.concat(parent_path[j], file))
      images[count] = image_data
      count = count + 1
   end
  end
  print('Loaded images:', #images)
 return images
end



return HN_mining