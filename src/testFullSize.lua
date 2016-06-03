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

local function toBBandAugm(input, winSize, stride)
  local map = input[1]
  local augmMap = input[2]
  local selection = torch.round(map)
  local numberWins = torch.sum(selection)
  local BB = torch.Tensor(numberWins, 4)
  local augm = torch.Tensor(numberWins,3)
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
          augm[count]:copy(torch.squeeze(augmMap[{{},{i},{j}}]))
          count = count + 1
        end
    end
  end   
  return BB, augm, scores
end

local function doBBregression(BB, augm, winSize)
    for i=1, BB:size(1) do
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
      p1[1] = p1[1] - sX*winSize
      p2[1] = p2[1] - sX*winSize
      -- Apply shift Y dim
      p1[2] = p1[2] - sY*winSize
      p2[2] = p2[2] - sY*winSize
      -- Apply offset
      local xoff = BB[i][1]
      local yoff = BB[i][2]
      BB[i][1] = xoff + p1[1] 
      BB[i][2] = yoff + p1[2]
      BB[i][3] = p2[1] + xoff
      BB[i][4] = p2[2] + yoff
    end
    return BB
end

local function rescaleBB(BB, scales, winSize)
  local offset = winSize/4
  for i=1, BB:size(1) do
    BB[i][1] = BB[i][1] + (offset - offset*scales[i]) 
    BB[i][2] = BB[i][2] + (offset - offset*scales[i])
    BB[i][3] = BB[i][3] - (offset - offset*scales[i])
    BB[i][4] = BB[i][4] - (offset - offset*scales[i])
  end
  return BB
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

local model = torch.load('MultiTaskModel.t7')
model:cuda()

local input = image.load('00000.ppm')
inputs = image.scale(input, "*1")

--local input = image.load()
local output = model:forward(inputs:cuda())
local BB, augm, scores
BB, augm, scores = toBBandAugm(output,48,8)



BBnms = nms(BB,0.3,scores)
BBsuppressed = BB:index(1, BBnms)
print(BBsuppressed)
vis_rois(inputs,BBsuppressed)

--BBnms = nms(BB,0.3,scores)
--BBsuppressed = BB:index(1, BBnms)
--Scalessuppressed = scales:index(1, BBnms)


--BBsuppressed = rescaleBB(BBsuppressed, Scalessuppressed, 48)


--vis_rois(inputs,BBsuppressed)

