local pl = (require 'pl.import_into')()

require 'torch'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local dataset = {}

dataset.pos_image_size = 48; 
dataset.neg_image_size = 48; 

-- Private function declaration
local generate_dataset
local loadPositiveTrainingSamples
local loadBackgroundTrainingImages
local extractPatches


-- These paths should not be changed
dataset.parent_path = "/usr/data/dneven/datasets/GTSDB/TrainIJCNN2013"
dataset.train_dataset_bin = "train_dataset.bin"
dataset.validation_dataset_bin = "validation_dataset.bin"
--dataset.test_dataset_bin = "test_dataset.bin"


-- This function will download the dataset in the './GTSRB' temp folder, and generate
-- binary files containing the dataset as torch tensors.
function dataset.generate_bin()
  if not pl.path.isfile(dataset.train_dataset_bin) then
    print('Generating bin of the dataset')
    local train_set = generate_dataset(dataset.parent_path)
    torch.save(dataset.train_dataset_bin, train_set)
    train_set = nil
    collectgarbage()

  end
end

-------------------------------------------------
-- Main Interface
-------------------------------------------------

-- Returns the train dataset
function dataset.get_train_dataset()
  dataset.generate_bin()
  local train_dataset = torch.load(dataset.train_dataset_bin)
  return train_dataset
end

-- Normalize the given dataset
-- You can specify the mean and std values, otherwise, they are computed on the given dataset
-- Return the mean and std values
function dataset.normalize_global(dataset, mean, std)
  local std = std or dataset.data:std()
  local mean = mean or dataset.data:mean()
  dataset.data:add(-mean)
  dataset.data:div(std)
  return mean, std
end

-- Locally normalize the dataset
function dataset.normalize_local(dataset)
  require 'image'
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  local batch = 200 -- Can be reduced if you experience memory issues
  local dataset_size = dataset.data:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset.data:narrow(1,i,local_batch))
    dataset.data:narrow(1,i,local_batch):copy(normalized_images)
  end
end

-------------------------------------------------
-- Private function
-------------------------------------------------

-- This will generate a dataset as torch tensor from a directory of images
-- parent_path is a string of the path containing all the images
-- use validation allows to generate a validation set
generate_dataset = function(parent_path)
  assert(parent_path, "A parent path is needed to generate the dataset")

  local posSamples, posLabels = loadPositiveTrainingSamples(parent_path,dataset.pos_image_size)
  local bgImages = loadBackgroundTrainingImages(parent_path)
  local negSamples = extractPatches(bgImages, 50, dataset.neg_image_size)
  
  local pos_data = torch.Tensor(#posSamples, 3,dataset.pos_image_size, dataset.pos_image_size)
  local pos_label = torch.Tensor(#posSamples, 1)
  local neg_data = torch.Tensor(#negSamples, 3, dataset.neg_image_size, dataset.neg_image_size)

  for i=1, #posSamples do
    pos_data[i]:copy(posSamples[i])
    pos_label[i] = posLabels[i]
  end
  
  for i=1, #negSamples do
    neg_data[i]:copy(negSamples[i])
  end

  main_dataset = {}
  main_dataset.pos_data = pos_data
  main_dataset.neg_data = neg_data
  main_dataset.pos_label = pos_label

  return main_dataset
end

loadPositiveTrainingSamples = function(parent_path, dim)
   print("loading pos training patches")
   local images = {}
   local labels = {}
   local csv_file_name = 'gt.csv'
   local csv_file_path = paths.concat(parent_path, csv_file_name)
   print(csv_file_path)
   local csv_content = pl.data.read(csv_file_path)
   for i = 1, #csv_content do
      local file = csv_content[i][1]
      local image_data = image.load(paths.concat(parent_path, file))
      local x1 = csv_content[i][2]
      local y1 = csv_content[i][3]
      local x2 = csv_content[i][4]
      local y2 = csv_content[i][5]
      local w = x2 - x1 + 1;
      local h = y2 - y1 + 1; 
      x1 = math.max(1, (x1 - w/2))
      x2 = math.min(1360, x2 + w/2)
      y1 = math.max(1, y1 - h/2)
      y2 = math.min(800, y2 + h/2)
      local patch = image_data[{{},{y1,y2},{x1,x2}}] 
      table.insert(images, image.scale(patch, dim, dim))
      table.insert(labels, csv_content[i][6])
      print("Im: " .. i)
      if(i%200 == 0) then
        collectgarbage()
      end
   end
   print('Loaded images:', #images)
   return images, labels
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

extractPatches = function(images, nPatchesImage, dim)
  local imDimX = images[1]:size(3)
  local imDimY = images[1]:size(2)
  local patches = {}
  for i = 1, #images do
    for j =  1, nPatchesImage do
      local x = math.random(1,imDimX - dim +1)
      local y = math.random(1,imDimY - dim +1)
      table.insert(patches, images[i][{{},{y, y+dim-1},{x, x+dim-1}}]:clone())
    end
  end
  return patches
end

-- Return the module
return dataset
