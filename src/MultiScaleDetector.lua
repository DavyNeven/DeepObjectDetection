require 'cudnn'
require 'image'
require 'src/nms'
require 'src/augmenter'

local MultiScaleDetector = {}

MultiScaleDetector.model = nil
MultiScaleDetector.winSize = 48
MultiScaleDetector.stride = 8
MultiScaleDetector.scales = {1,0.8,0.6,0.4}

-- Define private functions 
local toBBandAugm
local rescaleBB



-- Public functions 

function MultiScaleDetector.init(model, winSize, stride, scales)
  MultiScaleDetector.model = model
  MultiScaleDetector.winSize = MultiScaleDetector.winSize or winSize
  MultiScaleDetector.stride = MultiScaleDetector.stride or stride
  MultiScaleDetector.scales = MultiScaleDetector.scales or scales
end

function MultiScaleDetector.doMultiScaleDetection(im)
  -- Move image to GPU
  local BBs = {}
  local count = 1
  for i = 1, #MultiScaleDetector.scales do
    --print(im:size(1) .. " " .. im:size(2) .. " " .. im:size(3))
    local h = im:size(2)*MultiScaleDetector.scales[i]
    local w = im:size(3)*MultiScaleDetector.scales[i]
    local input = image.scale(im, w, h):cuda()
    local output = MultiScaleDetector.model:forward(input)
    local BB, augm = toBBandAugm(output, MultiScaleDetector.winSize, MultiScaleDetector.stride)
    -- Do NMS
    if(BB:dim() ~= 0) then
      local NMSselect = nms(BB,0.3,5)
      BB = BB:index(1, NMSselect)
      augm = augm:index(1, NMSselect)
      -- do BB regression
      undoRandomCrop(BB,augm,MultiScaleDetector.winSize)
      rescaleBB(BB, MultiScaleDetector.scales[i])
      BBs[count] = BB 
      count = count + 1
    end
  end
  -- Convert BBs to tensor
  local output = torch.Tensor()
  if(#BBs > 1) then
    local join = nn.JoinTable(1);
    output = join:forward(BBs)
    local NMSselect = nms(output,0.3,5)
    output = output:index(1, NMSselect)
  end
  return output
end

function MultiScaleDetector.doMultiScaleDetectionNoRegression(im)
  -- Move image to GPU
  local BBs = {}
  local count = 1
  for i = 1, #MultiScaleDetector.scales do
    --print(im:size(1) .. " " .. im:size(2) .. " " .. im:size(3))
    local h = im:size(2)*MultiScaleDetector.scales[i]
    local w = im:size(3)*MultiScaleDetector.scales[i]
    local input = image.scale(im, w, h):cuda()
    local output = MultiScaleDetector.model:forward(input)
    local BB, augm = toBBandAugm(output, MultiScaleDetector.winSize, MultiScaleDetector.stride)
    -- Do NMS
    if(BB:dim() ~= 0) then
      local NMSselect = nms(BB,0.3,5)
      BB = BB:index(1, NMSselect)
      rescaleBB(BB, MultiScaleDetector.scales[i])
      BBs[count] = BB 
      count = count + 1
    end
  end
  -- Convert BBs to tensor
  local output = torch.Tensor()
  if(#BBs > 1) then
    local join = nn.JoinTable(1);
    output = join:forward(BBs)
    local NMSselect = nms(output,0.3,5)
    output = output:index(1, NMSselect)
  end
  return output
end

-- private functions

rescaleBB = function(BB, scale)
  for i = 1, BB:size(1) do
    BB[i][1] = BB[i][1]*1/scale
    BB[i][2] = BB[i][2]*1/scale
    BB[i][3] = BB[i][3]*1/scale
    BB[i][4] = BB[i][4]*1/scale
  end
end

toBBandAugm = function(input, winSize, stride)
  local map = input[1]:squeeze()
  local augmMap = input[2]:squeeze()
  local selection = torch.round(map-0.2)
  local numberWins = torch.sum(selection)
  local BB = torch.Tensor(numberWins, 5)
  local augm = torch.Tensor(numberWins,3)
  local count = 1
  for i = 1, selection:size(1) do
    for j = 1, selection:size(2) do 
        if(selection[i][j] == 1) then
          local bb = torch.Tensor(5)
          bb[1] = (j-1)*stride + 1
          bb[2] = (i-1)*stride + 1
          bb[3] = bb[1] + winSize - 1
          bb[4] = bb[2] + winSize -1
          bb[5] = (map[i][j])
          BB[count]:copy(bb)
          augm[count]:copy(torch.squeeze(augmMap[{{},{i},{j}}]))
          count = count + 1
        end
    end
  end   
  return BB, augm
end

return MultiScaleDetector