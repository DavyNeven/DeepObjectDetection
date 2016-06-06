require 'image'
require 'cudnn'
require 'gnuplot'
require 'nms'

function vis_rois(img, rois, w)
    require 'qt'
    require 'image'
    require 'qtwidget'

    if not w then
        w = qtwidget.newwindow(img:size(3), img:size(2), '')
    end
    w:showpage()
    image.display({image = img, win = w})

    local roisTable = (type(rois) == 'table') and rois or {yellow = rois}
    for color, rois in pairs(roisTable) do
        for i = 1, rois:size(1) do
            local xmin, ymin, xmax, ymax = rois[i][1], rois[i][2], rois[i][3], rois[i][4]
            w:setcolor(color)
            w:setlinewidth(2)
            w:rectangle(xmin, ymin, xmax - xmin, ymax - ymin)
            w:stroke()
        end
    end
    return w
end

local function toBBandScale(input, winSize, stride)
  local map = input[1]
  local scaleMap = input[2]
  local selection = torch.round(map)
  local numberWins = torch.sum(selection)
  local BB = torch.Tensor(numberWins, 4)
  local scale = torch.Tensor(numberWins,1)
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
          scale[count] = (torch.squeeze(scaleMap[{{},{i},{j}}]))
          count = count + 1
        end
    end
  end   
  return BB, scale, scores
end

local function rescaleBB(BB, scales, winSize)
  scales = torch.squeeze(scales)
  local offset = winSize/4
  for i=1, BB:size(1) do
    BB[i][1] = BB[i][1] + (offset - offset*scales[i]) 
    BB[i][2] = BB[i][2] + (offset - offset*scales[i])
    BB[i][3] = BB[i][3] - (offset - offset*scales[i])
    BB[i][4] = BB[i][4] - (offset - offset*scales[i])
  end
  return BB
end

local model = torch.load('OnlyScaleModel05.t7')
model:cuda()

local input = image.load('00000.ppm')
input = image.scale(input, "*1")

--local input = image.load()
local output = model:forward(input:cuda())
local BB, scale, scores
BB, scale, scores = toBBandScale(output,48,8)
print(BB:size(1) == 0)
vis_rois(input, BB)
rescaleBB(BB, scale, 48)
local select = nms(BB,0.3,scores)
local BBnms = BB:index(1, select) 

vis_rois(input, BBnms)

