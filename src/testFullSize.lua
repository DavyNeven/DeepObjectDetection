require 'image'
require 'cudnn'
require 'cunn'
require 'gnuplot'
require 'nms'
require 'augmenter'

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
  print(input)
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

local model = torch.load('MultiTaskModel.t7')
model:cuda()
model:evaluate()

local im = image.load('00001.ppm')
local im1 = image.scale(im, "*0.8")
local im2 = image.scale(im, "*0.6")
local im3 = image.scale(im, "*0.4")

local input = {im1, im2, im3}

--local input = image.load()
for i=1, 3 do
    local output = model:forward(input[i]:cuda())
    print(output)
    local BB, augm
    BB, augm = toBBandAugm(output,48,8)
    print(BB[{{},5}])
    vis_rois(input[i], BB)
    
    
    -- Do NMS
    local NMSselect = nms(BB,0.3,5)
    BB = BB:index(1, NMSselect)
    augm = augm:index(1, NMSselect)
    print(BB[{{},5}])
    vis_rois(input[i], BB)
    
    -- do BB regression
    undoRandomCrop(BB,augm,48)
    vis_rois(input[i], BB)

end










