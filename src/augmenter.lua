require 'image'

-- requires images of size 48x48 with centered trafficSign
function randomScale(images, winSize)
  local scaledImages = torch.Tensor(images:size(1), images:size(2), images:size(3), images:size(4))
  local random = torch.rand(images:size(1) )
  for i=1, images:size(1) do
    local scaled = image.scale(images[i], "*" .. (1+random[i]))
    scaledImages[i] = image.crop(scaled, "c", winSize, winSize)
  end
  return scaledImages, random
end

-- Requires images of size 72x72 with centered trafficSign
function randomCrop(images, winSize)
local croppedImages = torch.Tensor(images:size(1), images:size(2), winSize, winSize)
local augm = torch.Tensor(images:size(1), 3)
for i=1, images:size(1) do
  local w = torch.floor(torch.rand(1)[1]*17) + 32 -- min 32, max 48
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
  w = (-24/26*(w-22)+48)/winSize
  x1 = (25 - x1)/winSize
  y1 = (25 - y1)/winSize
  augm[i] = torch.Tensor({w, x1, y1})
end
  return croppedImages, augm
end
