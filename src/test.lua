require 'cudnn'
require 'trainer'
require 'augmenter'

function rescaleImages(images, scales)
local scaledImages = torch.Tensor(images:size(1), images:size(2), images:size(3), images:size(4))
  for i=1, images:size(1) do
    local scaled = image.scale(images[i], "*" .. (2-scales[i]))
    scaledImages[i] = image.crop(scaled, "c", 48, 48)
  end
  return scaledImages
end

local data_provider = require 'dataset'


print("Loading training data...")
local train_dataset = data_provider.get_train_dataset()

print(cutorch.getMemoryUsage(1))
print("Loading multi-task model")
local model = torch.load('MultiTaskModel.t7')
print(cutorch.getMemoryUsage(1))
print("Move everyting to GPU")
model:cuda()

local pos_data = train_dataset.pos_data:narrow(1, 1, 100)


pos_data = randomScale(pos_data)

local output = model:forward(pos_data:cuda())


image.display({image=pos_data})
image.display({image=rescaleImages(pos_data,torch.squeeze(output[2]))})
