require 'cudnn'
require 'augmenter'

local data_provider = require 'dataset'

local function undoTransformation(images, augm, winSize)
  print(augm)
  local croppedImages = torch.Tensor(images:size(1), images:size(2), winSize, winSize)
  for i=1, images:size(1) do
    local val = augm[i]
    local s = val[1]
    local sX = val[2]
    local sY = val[3]
    local p1 = torch.Tensor({-1,-1})
    local p2 = torch.Tensor({1,1})
    -- Apply rescaling
    p1 = p1*s
    p2 = p2*s
    -- move to normal coordinates
    p1 = p1*winSize/2
    p2 = p2*winSize/2
    p1 = p1 + winSize/2
    p2 = p2 + winSize/2
    -- Apply shift X dim
    p1[1] = math.min(47, math.max(0, p1[1] - sX*winSize))
    p2[1] = math.min(47,math.max(p2[1] - sX*winSize))
    -- Appl shift Y dim
    p1[2] = math.min(47,math.max(0,p1[2] - sY*winSize))
    p2[2] = math.min(47,math.max(0,p2[2] - sY*winSize))
    print(p1)
    print(p2)
    local cropped = image.crop(images[i], p1[1], p1[2], p2[1], p2[2])
    croppedImages[i] = image.scale(cropped, winSize,winSize)
  end
  return croppedImages
end

print("Loading training data...")
local train_dataset = data_provider.get_train_dataset()

local pos_data = train_dataset.pos_data:narrow(1, 1, 100)
image.display({image=pos_data})
local cropped, augm_real = randomCrop(pos_data,48) 
image.display({image=cropped})

image.display({image=undoTransformation(cropped,augm_real,48)})



