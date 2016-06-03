local pl = (require 'pl.import_into')()

require 'nms'
require 'cunn'

local HN_mining = {}

local loadBackgroundTrainingImages
local getBackgroundTrainingImages
local mapToBB
local extractPatches

HN_mining.parent_path = "/usr/data/dneven/datasets/GTSDB/TrainIJCNN2013"
HN_mining.bg_bin = "bg_training_dataset.bin"

function HN_mining.get_hard_negatives(model, winSize, stride)
  local bg_images = getBackgroundTrainingImages()
  local totalPatches = {}
  for i=1, #bg_images do
    local output = model:forward(bg_images[i]:cuda())
    output = output[1]
    local BB, scores = mapToBB(output, winSize, stride)
    if(scores:nDimension() ~= 0) then 
      local selection = nms(BB,0.3,scores)
      BB = BB:index(1, selection)
      print("Image : " .. i .. " extracted patches: " .. BB:size(1))
      local patches = extractPatches(bg_images[i], BB, winSize)
      table.insert(totalPatches,patches)
    end
  end
  m = nn.JoinTable(1);
  local output = m:forward(totalPatches)
  return output  
end

extractPatches = function(image, BB, winSize)
  local patches = torch.Tensor(BB:size(1), 3, winSize, winSize)
  for i = 1, BB:size(1) do
      local x1 = BB[i][1]
      local y1 = BB[i][2]
      local x2 = BB[i][3]
      local y2 = BB[i][4]
      patches[i]:copy(image[{{},{y1, y2},{x1, x2}}])
  end
  return patches
end

mapToBB = function(map, winSize, stride)
 local selection = torch.round(map)
  local numberWins = torch.sum(selection)
  local BB = torch.Tensor(numberWins, 4)
  local scores = torch.Tensor(numberWins)
  local count = 1
  for i = 1, selection:size(2) do
    for j = 1, selection:size(3) do 
        if(selection[1][i][j] == 1) then
          local bb = torch.Tensor(4)
          bb[1] = (j-1)*stride + 1
          bb[2] = (i-1)*stride + 1
          bb[3] = bb[1] + winSize - 1
          bb[4] = bb[2] + winSize -1
          scores[count] = (map[1][i][j])
          BB[count]:copy(bb)
          count = count + 1
        end
    end
  end   
  return BB, scores
end

getBackgroundTrainingImages = function()
  local bg_images
  if not pl.path.isfile(HN_mining.bg_bin) then
    print('Generating bin of the background dataset')
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
 local csv_file_name = 'bg_training.csv'
 local csv_file_path = paths.concat(parent_path, csv_file_name)
 print(csv_file_path)
 local csv_content = pl.data.read(csv_file_path)
 for i = 1, #csv_content do
    local file = csv_content[i][1]
    local image_data = image.load(paths.concat(parent_path, file))
    images[i] = image_data
 end
 print('Loaded images:', #images)
 return images
end



return HN_mining