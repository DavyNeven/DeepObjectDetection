require 'image'
require 'gnuplot'

local path = '/esat/larimar/dneven/datasets/GTSDB/FullIJCNN2013'
local dataset = (require 'getGTSDB')()
local multiScaleDetector = require 'src/MultiScaleDetector'

local model = torch.load('MultiTaskModel.t7')
model:cuda()
model:evaluate()

multiScaleDetector.init(model)
local w
function displayImage(img, gt, preds)
  require 'qt'
  require 'qtwidget'
  if not w then
    w = qtwidget.newwindow(img:size(3), img:size(2), '')
  end
  w:showpage()
  image.display({image = img, win = w})

  w:setcolor('green')
  w:setlinewidth(2)
  for i = 1, #gt do
    w:rectangle(gt[i][1], gt[i][2], gt[i][3] - gt[i][1], gt[i][4] - gt[i][2])
  end
  w:stroke()

  w:setcolor('blue')
  w:setlinewidth(2)
  if(preds:dim()~=0 ) then
    for i = 1, preds:size(1) do
      w:rectangle(preds[i][1], preds[i][2], preds[i][3] - preds[i][1], preds[i][4] - preds[i][2])
    end
    w:stroke()
  end

  --    local roisTable = (type(rois) == 'table') and rois or {red = rois}
  --    for color, rois in pairs(roisTable) do
  --        for i = 1, rois:size(1) do
  --            local xmin, ymin, xmax, ymax = rois[i][1], rois[i][2], rois[i][3], rois[i][4]
  --            w:setcolor(color)
  --            w:setlinewidth(2)
  --            w:rectangle(xmin, ymin, xmax - xmin, ymax - ymin)
  --            w:stroke()
  --        end
  --    end
  --return w
end



local function getTP(preds, gt)
  local TP = 0
  if(preds:dim() ~= 0) then
    for j = 1, preds:size(1) do
      local pred = preds[j]
      local area1 = (pred[3] - pred[1]) * (pred[4] - pred[2])
      for i = 1, #gt do
        local gt = gt[i]
        local area2 = (gt[3] - gt[1]) * (gt[4] - gt[2])
        local x1 = math.max(pred[1], gt[1])
        local y1 = math.max(pred[2], gt[2])
        local x2 = math.min(pred[3], gt[3])
        local y2 = math.min(pred[4], gt[4])
        local inter = (x2 - x1)*(y2 - y1)
        --print(inter)
        local union = area1 + area2 - inter
        --print(union)
        local IoU = inter/union
        --print(IoU)
        if(IoU > 0.5) then
          TP = TP + 1
          break
        end
      end
    end
  end
  local FP = 0
  if(preds:dim() ~= 0) then
    FP = preds:size(1) - TP
  end
  return TP, FP
end


local totalGT = 0
local totalTP = 0
local totalPreds = 0 
for i = 600, 700 do
  local im_path = paths.concat(path, dataset[i][1])
  local im = image.load(im_path)
  local gt = dataset[i][2]
  local preds = multiScaleDetector.doMultiScaleDetection(im)
  --print(preds)
  --print(gt)
  local TP, FP = getTP(preds, gt)
  totalTP = totalTP + TP
  totalGT = totalGT + #gt
  if(preds:dim() ~= 0) then
    totalPreds = totalPreds + preds:size(1)
  end
  print('TP: ' .. TP)
  print('FP: ' .. FP)
  displayImage(im, gt, preds)
  --io.read()
end

print('totalGT: ' .. totalGT)
print('totalTP: ' .. totalTP)
print('totalPreds: ' .. totalPreds)
print('recall: ' .. totalTP/totalGT)
print('precision: ' .. totalTP/totalPreds)
