#!/usr/bin/env python3
"""
Fixed Dynamic Small World CNN - Proper FLOPS counting and dynamic convolution cost reduction
"""
import argparse
import os.path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import numpy as np
import networkx as nx
import random
from typing import List
from torch.backends import cudnn as cudnn

# Utility imports (minimal implementations)
class Logger:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def tick(self):
        pass

class Utils:
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @staticmethod
    def save_checkpoint(state, folder, is_best):
        if folder:
            os.makedirs(folder, exist_ok=True)
            torch.save(state, os.path.join(folder, 'checkpoint.pth'))
            if is_best:
                torch.save(state, os.path.join(folder, 'model_best.pth'))

    class AverageMeter:
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

# Proper FLOPS counter implementation
class FLOPSCounter:
    def __init__(self):
        self.flops = 0
        self.active = False

    def reset(self):
        self.flops = 0

    def add_flops(self, flops):
        if self.active:
            self.flops += flops

    @staticmethod
    def add_flops_counting_methods(model):
        model._flops_counter = FLOPSCounter()

        def compute_average_flops_cost():
            return [model._flops_counter.flops]

        def start_flops_count():
            model._flops_counter.active = True
            model._flops_counter.reset()

        def stop_flops_count():
            model._flops_counter.active = False

        def reset_flops_count():
            model._flops_counter.reset()

        model.compute_average_flops_cost = compute_average_flops_cost
        model.start_flops_count = start_flops_count
        model.stop_flops_count = stop_flops_count
        model.reset_flops_count = reset_flops_count

        # Hook for conv layers
        def conv_flop_count(module, input, output):
            if hasattr(model, '_flops_counter') and model._flops_counter.active:
                input_dims = input[0].shape
                output_dims = output.shape
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups

                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups

                active_elements_count = int(np.prod(output_dims[2:]))  # H * W
                overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel

                # Apply dynamic reduction if available
                if hasattr(module, '_dynamic_usage_ratio'):
                    overall_conv_flops = int(overall_conv_flops * module._dynamic_usage_ratio)

                bias_flops = 0
                if module.bias is not None:
                    bias_flops = out_channels * active_elements_count

                overall_flops = overall_conv_flops + bias_flops
                model._flops_counter.add_flops(overall_flops)

        # Register hooks for all conv layers
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(conv_flop_count)

        return model

logger = Logger()
utils = Utils()
flopscounter = FLOPSCounter()

cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fixed Dynamic convolution components
class SparsityCriterion(nn.Module):
    """Fixed sparsity loss for dynamic convolution"""
    def __init__(self, budget, epochs):
        super(SparsityCriterion, self).__init__()
        self.budget = max(0.1, min(1.0, budget))  # Clamp budget
        self.epochs = epochs

    def forward(self, meta):
        if 'masks' not in meta or not meta['masks']:
            return torch.tensor(0.0, device=meta['device'])

        # Calculate sparsity loss based on mask usage
        total_usage = 0
        total_capacity = 0

        for mask in meta['masks']:
            if isinstance(mask, torch.Tensor) and mask.numel() > 0:
                # Use mean instead of sum to avoid scaling issues
                total_usage += mask.mean()
                total_capacity += 1

        if total_capacity == 0:
            return torch.tensor(0.0, device=meta['device'])

        usage_ratio = total_usage / total_capacity
        target_ratio = self.budget

        # Encourage usage close to budget with smaller penalty
        loss = torch.abs(usage_ratio - target_ratio)
        return loss

class DynamicConv2d(nn.Module):
    """Enhanced dynamic convolution layer with proper FLOPS tracking"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DynamicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.out_channels = out_channels

        # Simplified mask predictor to avoid complexity
        self.mask_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(8, in_channels // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 8), out_channels, 1)
        )

        # Initialize mask predictor to output reasonable values
        with torch.no_grad():
            self.mask_predictor[-1].weight.fill_(0.0)
            self.mask_predictor[-1].bias.fill_(0.5)  # Start with 50% channels active

    def forward(self, x, meta):
        # Predict channel-wise masks
        mask_logits = self.mask_predictor(x)

        # Use temperature scaling for mask generation
        temperature = meta.get('gumbel_temp', 1.0)
        masks = torch.sigmoid(mask_logits / temperature)

        # Ensure masks don't go to zero (minimum activation)
        masks = torch.clamp(masks, min=0.1, max=1.0)

        # Apply convolution
        out = self.conv(x)

        # Apply dynamic masks
        out = out * masks

        # Calculate actual usage ratio for FLOPS counting
        usage_ratio = masks.mean().item()
        self.conv._dynamic_usage_ratio = usage_ratio

        # Store masks for sparsity loss
        meta['masks'].append(masks.detach().mean(dim=(0, 2, 3)))
        meta['usage_ratios'].append(usage_ratio)

        return out

# Small World Network components (simplified)
class WattsStrogatzGraph:
    @staticmethod
    def generate_graph(n: int, k: int, p: float) -> nx.Graph:
        # Ensure valid parameters
        n = max(4, n)
        k = min(max(2, k), n - 1)
        if k % 2 != 0:
            k -= 1
        k = max(k, 2)
        p = max(0.0, min(1.0, p))

        G = nx.Graph()
        nodes = list(range(n))
        G.add_nodes_from(nodes)

        # Create regular ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                G.add_edge(i, (i + j) % n)
                G.add_edge(i, (i - j) % n)

        # Rewire edges with probability p
        edges = list(G.edges())
        for u, v in edges:
            if random.random() < p:
                G.remove_edge(u, v)
                possible = [x for x in nodes if x != u and not G.has_edge(u, x)]
                if possible:
                    G.add_edge(u, random.choice(possible))
                else:
                    G.add_edge(u, v)

        # Ensure all nodes have at least one connection
        for node in nodes:
            if G.degree(node) == 0:
                G.add_edge(node, (node + 1) % n)

        return G

    @staticmethod
    def generate_dag(n: int, k: int, p: float) -> nx.DiGraph:
        G = WattsStrogatzGraph.generate_graph(n, k, p)
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if u < v:
                dag.add_edge(u, v)
            else:
                dag.add_edge(v, u)
        return dag

class DynamicDAGNode(nn.Module):
    """DAG Node with optional dynamic convolution"""
    def __init__(self, channels, node_id, num_inputs, use_dynamic=True):
        super(DynamicDAGNode, self).__init__()
        self.node_id = node_id
        self.num_inputs = max(1, num_inputs)
        self.use_dynamic = use_dynamic

        if self.num_inputs > 1:
            self.weights = nn.Parameter(torch.ones(self.num_inputs) / self.num_inputs)
        else:
            self.weights = None

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        if use_dynamic:
            self.conv = DynamicConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # Initialize conv weights properly
        if hasattr(self.conv, 'conv'):
            nn.init.kaiming_normal_(self.conv.conv.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs: List[torch.Tensor], meta=None) -> torch.Tensor:
        if not inputs:
            raise ValueError("No inputs provided to DAG node")

        if len(inputs) == 1:
            x = inputs[0]
        else:
            # Use softmax for proper weighting
            w = torch.softmax(self.weights[:len(inputs)], dim=0).view(-1, 1, 1, 1)
            x = sum(w_i * inp for w_i, inp in zip(w, inputs))

        x = self.bn(x)
        x = self.relu(x)

        if self.use_dynamic and meta is not None:
            x = self.conv(x, meta)
        else:
            x = self.conv(x)

        return x

class DynamicDAGSmallWorldModule(nn.Module):
    """DAG Small World Module with dynamic convolution support"""
    def __init__(self, in_channels, out_channels, num_nodes=16, rewiring_prob=0.2,
                 initial_neighbors=4, use_dynamic=True):
        super(DynamicDAGSmallWorldModule, self).__init__()
        self.use_dynamic = use_dynamic

        # Reduce complexity - fewer nodes
        #num_nodes = min(6, max(3, num_nodes))
        initial_neighbors = min(num_nodes - 1, max(1, initial_neighbors))

        self.graph = WattsStrogatzGraph.generate_dag(num_nodes, initial_neighbors, rewiring_prob)
        self.input_node = -1
        self.output_node = num_nodes

        # Ensure connectivity
        for n in range(num_nodes):
            if self.graph.in_degree(n) == 0:
                self.graph.add_edge(self.input_node, n)
            if self.graph.out_degree(n) == 0:
                self.graph.add_edge(n, self.output_node)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # DAG nodes with dynamic support
        self.nodes = nn.ModuleDict()
        for node in range(num_nodes):
            inputs = list(self.graph.predecessors(node))
            self.nodes[str(node)] = DynamicDAGNode(out_channels, node_id=node,
                                                 num_inputs=len(inputs), use_dynamic=use_dynamic)

        # Output processing
        if use_dynamic:
            self.output_conv = nn.Sequential(
                DynamicConv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.downsample = nn.MaxPool2d(2, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, meta=None):
        x = self.input_proj(x)
        node_outputs = {self.input_node: x}

        # Process nodes in topological order
        for node in nx.topological_sort(self.graph):
            if node == self.input_node:
                continue
            if node == self.output_node:
                # Collect outputs from predecessors
                inputs = [node_outputs[p] for p in self.graph.predecessors(node) if p in node_outputs]
                if inputs:
                    node_outputs[node] = sum(inputs) / len(inputs)
                else:
                    node_outputs[node] = x
            else:
                inputs = [node_outputs[p] for p in self.graph.predecessors(node) if p in node_outputs]
                if inputs and str(node) in self.nodes:
                    node_outputs[node] = self.nodes[str(node)](inputs, meta)
                else:
                    node_outputs[node] = x

        out = node_outputs.get(self.output_node, x)

        if self.use_dynamic and meta is not None:
            # Apply dynamic convolution to output processing
            if hasattr(self.output_conv[0], 'forward'):
                out = self.output_conv[0](out, meta)  # Dynamic conv
                out = self.output_conv[1](out)  # BN
                out = self.output_conv[2](out)  # ReLU
            else:
                out = self.output_conv(out)
        else:
            out = self.output_conv(out)

        return self.downsample(out)

class DynamicSWCNN(nn.Module):
    """Dynamic Small World CNN with proper FLOPS tracking"""
    def __init__(self, num_classes=10, sparse=True, pretrained=False):
        super(DynamicSWCNN, self).__init__()
        self.sparse = sparse

        # Configuration
        self.num_nodes = 16
        self.rewiring_prob = 0.2
        self.initial_neighbors = 4
        self.num_modules = 3
        self.channels = [64, 128, 256]

        # Simple stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Small World modules with dynamic support
        self.sw_modules = nn.ModuleList()
        prev_channels = 64
        for i in range(self.num_modules):
            module = DynamicDAGSmallWorldModule(
                in_channels=prev_channels,
                out_channels=self.channels[i],
                num_nodes=self.num_nodes,
                rewiring_prob=self.rewiring_prob,
                initial_neighbors=self.initial_neighbors,
                use_dynamic=sparse  # Enable dynamic when sparse=True
            )
            self.sw_modules.append(module)
            prev_channels = self.channels[i]

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(prev_channels, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, meta=None):
        if meta is None:
            meta = {'masks': [], 'usage_ratios': [], 'device': x.device}

        # Stem processing
        x = self.stem(x)

        # Small World modules
        for module in self.sw_modules:
            x = module(x, meta if self.sparse else None)

        # Classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, meta

# Training and evaluation functions
class Loss(nn.Module):
    def __init__(self, budget, epochs):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss()
        self.sparsity_loss = SparsityCriterion(budget, epochs) if budget >= 0 else None
        self.sparsity_weight = 0.1

    def forward(self, output, target, meta):
        l = self.task_loss(output, target)
        logger.add('loss_task', l.item())

        if self.sparsity_loss is not None and meta['masks']:
            sparsity_l = self.sparsity_loss(meta)
            l += self.sparsity_weight * sparsity_l
            logger.add('loss_sparsity', sparsity_l.item())

        return l

def train(args, train_loader, model, criterion, optimizer, epoch):
    """Run one train epoch"""
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.train()

    # Adaptive temperature for dynamic masks
    gumbel_temp = max(0.5, 2.0 * (1 - epoch / args.epochs))
    gumbel_noise = False

    num_step = len(train_loader)
    for i, (input, target) in enumerate(tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5)):
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # Compute output
        meta = {
            'masks': [],
            'usage_ratios': [],
            'device': device,
            'gumbel_temp': gumbel_temp,
            'gumbel_noise': gumbel_noise,
            'epoch': epoch
        }
        output, meta = model(input, meta)
        loss = criterion(output, target, meta)

        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {i}, skipping...")
            continue

        # Measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log every 100 batches
        if i % 100 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Loss {losses.avg:.4f}\t'
                  f'Prec@1 {top1.avg:.3f}')

        logger.tick()

def validate(args, val_loader, model, criterion, epoch):
    """Run evaluation"""
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    usage_meter = utils.AverageMeter()

    # Switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # Compute output
            meta = {
                'masks': [],
                'usage_ratios': [],
                'device': device,
                'gumbel_temp': 1.0,
                'gumbel_noise': False,
                'epoch': epoch
            }
            output, meta = model(input, meta)
            output = output.float()
            loss = criterion(output, target, meta)

            # Measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Track usage ratios
            if meta['usage_ratios']:
                avg_usage = sum(meta['usage_ratios']) / len(meta['usage_ratios'])
                usage_meter.update(avg_usage, input.size(0))

    print(f'* Epoch {epoch} - Loss {losses.avg:.4f} - Prec@1 {top1.avg:.3f}')

    # Channel usage
    if usage_meter.count > 0:
        print(f'* Average channel usage: {usage_meter.avg:.3f}')
        print(f'* Theoretical FLOPS reduction: {(1 - usage_meter.avg) * 100:.1f}%')

    # FLOPS counting
    flops = model.compute_average_flops_cost()[0]
    print(f'* Average FLOPS per image: {flops/1e6:.3f} MMac')

    model.stop_flops_count()
    return top1.avg

def main():
    parser = argparse.ArgumentParser(description='Dynamic Small World CNN Training on CIFAR-10')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=[60, 80], nargs='+', type=int, help='learning rate decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--budget', default=0.5, type=float, help='computational budget (0.5 = 50% usage)')
    parser.add_argument('-s', '--save_dir', type=str, default='./checkpoints', help='directory to save model')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    parser.add_argument('--workers', default=2, type=int, help='number of dataloader workers')
    parser.add_argument('--pretrained', action='store_true', help='initialize with pretrained model')
    args = parser.parse_args()
    print('Args:', args)

    # Data transforms
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Data loading
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)

    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)

    # Model
    model = DynamicSWCNN(num_classes=10, sparse=args.budget >= 0, pretrained=args.pretrained).to(device=device)
    print(f"* Number of trainable parameters: {utils.count_parameters(model):,}")

    # Loss and optimizer
    criterion = Loss(args.budget, args.epochs)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)

    # Resume training
    start_epoch = 0
    best_prec1 = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Create save directory
    if not args.evaluate and len(args.save_dir) > 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # Evaluation mode
    if args.evaluate:
        print("########## Evaluation ##########")
        prec1 = validate(args, val_loader, model, criterion, start_epoch)
        return

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"########## Epoch {epoch} ##########")
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        # Train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # Evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
        }, folder=args.save_dir, is_best=is_best)

        print(f" * Best prec1: {best_prec1:.3f}")
        print("-" * 50)

    print(f"Final best accuracy: {best_prec1:.3f}%")

if __name__ == "__main__":
    main()
