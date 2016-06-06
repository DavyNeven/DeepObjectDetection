require 'cudnn'
require 'trainer'
require 'augmenter'
require 'optim'
paths.dofile('Optim.lua')

local HN = require 'HN_mining'


local data_provider = require 'dataset'
local network_provider = require 'networks'

print("Loading training data...")
local train_dataset = data_provider.get_train_dataset()

print("Loading multi-task model")
local model = network_provider.getMultiTaskModel()

print("Creating multi-task criterion")
local criterion = nn.ParallelCriterion()
criterion:add(nn.BCECriterion())
criterion:add(nn.BCECriterion())

print("Creating SGD optimizer")
local optimState = {learningRate = 0.01, momentum = 0.9, weightDecay = 5e-4}
local optimizer = nn.Optim(model, optimState)

print("Move everyting to GPU")
model:cuda()
criterion:cuda()

-- Define batchSize
local batchSize = 1

for epoch=1,80 do
   print("epoch: " .. epoch)
   train(model,criterion,optimizer,train_dataset,batchSize)
end

test(model,criterion,train_dataset,batchSize)

-- Do hard negative mining
local hn_patches = HN.get_hard_negatives(model,48,8)
print("#HN-patches: " .. hn_patches:size(1))

-- Combine negative patches with HN patches
m = nn.JoinTable(1);
train_dataset.neg_data = m:forward({train_dataset.neg_data, hn_patches})


for epoch=1,80 do
   print("epoch: " .. epoch)
   train(model,criterion,optimizer,train_dataset,batchSize)
end

test(model,criterion,train_dataset,batchSize)

-- Do hard negative mining
local hn_patches = HN.get_hard_negatives(model,48,8)
print("#HN-patches: " .. hn_patches:size(1))

-- Combine negative patches with HN patches
m = nn.JoinTable(1);
train_dataset.neg_data = m:forward({train_dataset.neg_data, hn_patches})

for epoch=1,80 do
   print("epoch: " .. epoch)
   train(model,criterion,optimizer,train_dataset,batchSize)
end

test(model,criterion,train_dataset,batchSize)

-- Do hard negative mining
local hn_patches = HN.get_hard_negatives(model,48,8)
print("#HN-patches: " .. hn_patches:size(1))

torch.save('OnlyScaleModel05.t7', model)


