require 'cudnn'

local networks = {}

function networks.getMultiTaskModel()

  local net = nn.Sequential()
  --net = nn.Sequential()
  net:add(cudnn.SpatialConvolution(3, 100, 3, 3)) -- 3 input image channels, 100 output channels, 3x3 convolution kernel
  net:add(cudnn.ReLU(true))                       -- non-linearity 
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))       -- A max-pooling operation that looks at 2x2 windows and finds the max.
  
  net:add(cudnn.SpatialConvolution(100, 150, 4, 4)) -- 3 input image channels, 100 output channels, 3x3 convolution kernel
  net:add(cudnn.ReLU(true))                       -- non-linearity 
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))       -- A max-pooling operation that looks at 2x2 windows and finds the max.
  
  net:add(cudnn.SpatialConvolution(150, 250, 3, 3)) -- 3 input image channels, 100 output channels, 3x3 convolution kernel
  net:add(cudnn.ReLU(true))                       -- non-linearity 
  net:add(cudnn.SpatialMaxPooling(2,2,2,2))       -- A max-pooling operation that looks at 2x2 windows and finds the max.
    
  -- shared fully connected layer
  net:add(cudnn.SpatialConvolution(250, 200, 4, 4))
  net:add(cudnn.ReLU(true))
  net:add(nn.Dropout())  
    
  local pr1 = nn.ConcatTable()
  local b1 = nn.Sequential()
  -- Fully connected layers
  b1:add(cudnn.SpatialConvolution(200, 1, 1, 1))
  b1:add(cudnn.Sigmoid())
  local b2 = nn.Sequential()
  -- Fully connected layers
  b2:add(cudnn.SpatialConvolution(200, 1, 1, 1))
  b2:add(cudnn.Sigmoid())
  pr1:add(b1)
  pr1:add(b2)
  
  net:add(pr1)
  
  return net

end

return networks