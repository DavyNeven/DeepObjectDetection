require 'image'

-- requires images of size 48x48 with centered trafficSign
function randomScale(images)
  local scaledImages = torch.Tensor(images:size(1), images:size(2), images:size(3), images:size(4))
  local random = torch.rand(images:size(1) )
  for i=1, images:size(1) do
    local scaled = image.scale(images[i], "*" .. (1+random[i]))
    scaledImages[i] = image.crop(scaled, "c", 48, 48)
  end
  return scaledImages, random
end

-- Requires images of size 72x72 with centered trafficSign
function randomCrop(images, winSize)
local croppedImages = torch.Tensor(images:size(1), images:size(2), winSize, winSize)
local augm = torch.Tensor(images:size(1), 3)
for i=1, images:size(1) do
  local w = torch.floor(torch.rand(1)[1]*27) + 22 -- min 22, max 48
  local shiftX = 0--torch.trunc((torch.rand(1)[1]*2 - 1)*(w-22)/2)
  local shiftY = 0--torch.trunc((torch.rand(1)[1]*2 - 1)*(w-22)/2)
  local cx = 36 + shiftX
  local cy = 36 + shiftY
  local x1 = torch.round(cx-w/2)
  local y1 = torch.round(cy-w/2)
  local x2 = torch.round(cx+w/2)
  local y2 = torch.round(cy+w/2)
  local res = image.crop(images[i], x1, y1, x2, y2)
  croppedImages[i] = image.scale(res, winSize,winSize)
  local scale = 1 - 0.5/(48-22)*(w - 22)
  shiftX = shiftX/images:size(3)
  shiftY = shiftY/images:size(3)
  augm[i] = torch.Tensor({scale, shiftX, shiftY})
end
  return croppedImages, augm
end
