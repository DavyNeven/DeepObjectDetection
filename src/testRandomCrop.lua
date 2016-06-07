require 'cudnn'
require 'augmenter'

local data_provider = require 'dataset'

local function undoTransformation(images, augm, winSize)
  print(augm)
  local croppedImages = torch.Tensor(images:size(1), images:size(2), winSize, winSize)
  for i=1, images:size(1) do
    local val = augm[i]
    print(val)
    local w = val[1]*winSize
    local x1 = val[2]*winSize
    local y1 = val[3]*winSize
    print(x1 .. " " .. y1 .. " " .. w)
    local cropped = image.crop(images[i], x1, y1, math.max(x1+1, math.min(47,x1+w)),math.max(y1+1, math.min(47, y1+w)))
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

local multiTaskModel = torch.load("MultiTaskModel.t7")
local output = multiTaskModel:forward(cropped:cuda())

image.display({image=undoTransformation(cropped,torch.squeeze(output[2]),48)})
