require 'cudnn'
require 'image'
require 'gnuplot'

local MultiScaleDetector = require 'MultiScaleDetector'


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


-- Load model
local model = torch.load('MultiTaskModel.t7')
model:cuda()
model:evaluate()

MultiScaleDetector.init(model)
local im = image.load("00871.ppm")
local BBs = MultiScaleDetector.doMultiScaleDetection(im)
vis_rois(im, BBs)