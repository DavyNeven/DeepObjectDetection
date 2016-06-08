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

    local roisTable = (type(rois) == 'table') and rois or {red = rois}
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

local function toBBandAugm(input, winSize, stride)
  local map = input[1]
  local augmMap = input[2]
  local selection = torch.round(map)
  local numberWins = torch.sum(selection)
  local BB = torch.Tensor(numberWins, 5)
  local augm = torch.Tensor(numberWins,3)
  local count = 1
  for i = 1, selection:size(2) do
    for j = 1, selection:size(3) do 
        if(selection[1][i][j] == 1) then
          local bb = torch.Tensor(5)
          bb[1] = (j-1)*stride + 1
          bb[2] = (i-1)*stride + 1
          bb[3] = bb[1] + winSize - 1
          bb[4] = bb[2] + winSize -1
          bb[5] = (map[1][i][j])
          BB[count]:copy(bb)
          augm[count]:copy(torch.squeeze(augmMap[{{},{i},{j}}]))
          count = count + 1
        end
    end
  end   
  return BB, augm
end

local function doBBregression(BB, augm, winSize)
  for i=1, BB:size(1) do
        local val = augm[i]
        local w = val[1]*winSize
        local x1 = val[2]*winSize
        local y1 = val[3]*winSize
        BB[i][1] = BB[i][1] + x1
        BB[i][2] = BB[i][2] + y1
        BB[i][3] = BB[i][1] + w
        BB[i][4] = BB[i][2] + w
  end
  return BB
end

local model = torch.load('MultiTaskModel.t7')
model:cuda()

local input = image.load('00829.ppm')
inputs = image.scale(input, "*1")

--local input = image.load()
local output = model:forward(inputs:cuda())
local BB, augm
BB, augm = toBBandAugm(output,48,8)

vis_rois(inputs, BB)

-- Do NMS
local NMSselect = nms(BB,0.3,5)
BB = BB:index(1, NMSselect)
augm = augm:index(1, NMSselect)

vis_rois(inputs, BB)

-- do BB regression
doBBregression(BB,augm,48)
vis_rois(inputs, BB)






