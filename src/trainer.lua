require 'augmenter'

function train(model, criterion,optimizer, train_dataset, batchSize)

  local nPosSamples = train_dataset.pos_data:size(1)
  print('nPosSamples: ' .. nPosSamples)
  local nNegSamples = train_dataset.neg_data:size(1)
  print('nNegSamples: ' .. nNegSamples)
  
  local nBatches = torch.floor(nPosSamples/batchSize)
  

  print("Shuffling data")
  local shuffle = torch.randperm(nPosSamples):type('torch.LongTensor')
  local pos_data = train_dataset.pos_data:index(1,shuffle)
  shuffle = torch.randperm(nNegSamples):type('torch.LongTensor')
  local neg_data = train_dataset.neg_data:index(1, shuffle)

  model:training()
  local trainError = 0
  local augm
  for batchidx = 1, nBatches do
      local pos_input = pos_data:narrow(1, (batchidx-1)*batchSize + 1, batchSize)
      pos_input, augm = randomCrop(pos_input,48)
      local neg_input = neg_data:narrow(1, (batchidx-1)*batchSize + 1, batchSize)
      local inputs = torch.cat(pos_input, neg_input, 1)
      local labels = torch.cat(torch.ones(batchSize), torch.zeros(batchSize))
      err = optimizer:optimize(optim.sgd, inputs:cuda(), {labels:cuda(), torch.cat(augm, torch.zeros(batchSize,3)):cuda()}, criterion)
      --print('epoch : ', epoch, 'batch : ', batchidx, 'train error : ', err)
      trainError = trainError + err
   end
   print('trainError : ', trainError / nBatches)

end



function test(model, criterion,train_dataset, batchSize)

  local nPosSamples = train_dataset.pos_data:size(1)
  print('nPosSamples: ' .. nPosSamples)
  local nNegSamples = train_dataset.neg_data:size(1)
  print('nNegSamples: ' .. nNegSamples)
  
  local nBatches = torch.floor(nPosSamples/batchSize)


   model:evaluate()
   local valError = 0
   local correct = 0
   local all = 0
   local augm
   for batchidx = 1, nBatches do
      local pos_input = train_dataset.pos_data:narrow(1, (batchidx-1)*batchSize + 1, batchSize)
      pos_input, augm = randomCrop(pos_input,48)
      local labels = torch.ones(batchSize)
      local pred = model:forward(pos_input:cuda())
      valError = valError + criterion:forward(pred, {labels:cuda(), augm:cuda()})
      preds = torch.round(pred[1])
      correct = correct + preds:eq(labels[1]):sum()
      all = all + preds:size(1)
   end

   print('train pos error : ', valError / nBatches)
   print('train pos accuracy % : ', correct / all * 100)

   local valError = 0
   local correct = 0
   local all = 0
   for batchidx = 1, nBatches do
      local neg_input = train_dataset.neg_data:narrow(1, (batchidx-1)*batchSize + 1, batchSize)
      local labels = {torch.zeros(batchSize):cuda(), torch.zeros(batchSize,3):cuda()}
      local pred = model:forward(neg_input:cuda())
      valError = valError + criterion:forward(pred, labels)
      preds = torch.round(pred[1])
      correct = correct + preds:eq(labels[1]:cuda()):sum()
      all = all + preds:size(1)
   end
   
   print('train neg error : ', valError / nBatches)
   print('train neg accuracy % : ', correct / all * 100)

end