"""Inspectible reference training, with a shared local checkpoint contract.

This compact trainer uses torchvision/PIL, not the historical DALI backend.
Stored visual diets (DVD/Statistical/Shaders) are supplied under data/.
"""
import copy
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import alexnet, resnet50
from PIL import Image
from . import paths

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def backbone(architecture="alexnet"):
    model=alexnet(weights=None) if architecture=="alexnet" else resnet50(weights=None)
    if architecture=="alexnet":
        for layer in model.features:
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight,0,math.sqrt(2/(layer.kernel_size[0]*layer.kernel_size[1]*layer.in_channels)))
                nn.init.zeros_(layer.bias)
    return model


def simclr_loss(z1,z2,temperature=.5):
    n=len(z1)
    if n<2:raise ValueError("SimCLR requires at least two images")
    z=torch.nn.functional.normalize(torch.cat([z1,z2]).float(),dim=1)
    logits=z@z.T/temperature;logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    return torch.nn.functional.cross_entropy(logits,(torch.arange(2*n,device=z.device)+n)%(2*n))


def dino_loss(student,teacher,center,student_temperature=.1,teacher_temperature=.04):
    """Cross-view teacher soft targets, excluding each matching global crop."""
    target=[torch.softmax((v-center)/teacher_temperature,dim=-1).detach() for v in teacher]
    terms=[-(q*torch.log_softmax(v/student_temperature,dim=-1)).sum(-1).mean()
           for i,q in enumerate(target) for j,v in enumerate(student) if i!=j]
    if not terms:raise ValueError("No cross-view DINO loss terms")
    return torch.stack(terms).mean()


def views_transform():
    return transforms.Compose([transforms.RandomResizedCrop(224,scale=(.08,1.)),
        transforms.RandomHorizontalFlip(),transforms.RandomApply([transforms.ColorJitter(.4,.4,.2,.1)],p=.8),
        transforms.RandomGrayscale(p=.2),transforms.RandomApply([transforms.GaussianBlur(23,sigma=(.1,2.))],p=.5),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


class Images(Dataset):
    def __init__(self,root,algorithm,seed,correct_fraction=1.):
        self.paths=sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png"))
        if not self.paths:raise FileNotFoundError("No images under the configured in-tree dataset")
        self.algorithm=algorithm
        classes=sorted({p.parent.name for p in self.paths});lookup={c:i for i,c in enumerate(classes)}
        self.labels=np.array([lookup[p.parent.name] for p in self.paths])
        self.retained=np.ones(len(self.paths),bool)
        if correct_fraction<1:
            rng=np.random.default_rng(seed);retained=rng.choice(len(self.paths),round(len(self.paths)*correct_fraction),replace=False)
            self.retained[:]=False;self.retained[retained]=True
            changed=np.flatnonzero(~self.retained)
            # Fixed label permutation may accidentally preserve some labels;
            # report the realized correspondence separately from retained fraction.
            self.labels[changed]=self.labels[rng.permutation(changed)]
        self.view=views_transform()
        self.basic=transforms.Compose([transforms.RandomResizedCrop(227),transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        self.local=transforms.Compose([transforms.RandomResizedCrop(96,scale=(.05,.4)),
                                      transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        resolved=self.paths[i].resolve()
        if not resolved.is_relative_to(paths.ROOT):raise ValueError("Dataset symlink escapes checkout")
        with Image.open(resolved) as image:
            image=image.convert("RGB")
            if self.algorithm=="simclr":return [self.view(image),self.view(image)],0
            if self.algorithm=="dino":return [self.view(image),self.view(image)]+[self.local(image) for _ in range(8)],0
            return self.basic(image),int(self.labels[i])


def training_plan(state,run):
    protocol=paths.protocol()
    seed=paths.run_seed(state,run)
    architecture="resnet50" if state.startswith("resnet50") else "alexnet"
    algorithm="simclr" if state in ["simclr","noise","grating","statistical","shaders"] else "rotnet" if state=="rotnet" else "dino" if state=="dino" else "supervised"
    dataset=state if state in ["noise","grating","statistical","shaders"] else "imagenet"
    if state=="dvd":dataset="dvd"
    correct={"label_shuffled":0.,"correct_05":.05,"correct_10":.1,"correct_20":.2,"correct_50":.5}.get(state,1.)
    return {"state":state,"run":run,"seed":seed,"architecture":architecture,"algorithm":algorithm,
            "expected_networks_in_state":len(paths.runs(state)),
            "dataset":paths.relative(paths.inside("data","dvd")) if dataset=="dvd" else paths.relative(paths.dataset(dataset)/("train" if dataset=="imagenet" else "")),
            "epochs":protocol["endpoint_epochs"].get(state,protocol["default_endpoint_epoch"]),
            "retained_correct_fraction":correct,"checkpoint":paths.relative(paths.checkpoint(state,run,protocol["endpoint_epochs"].get(state,protocol["default_endpoint_epoch"]))),
            "source":"reference trainer; historical performance not asserted"}


def train(state,run,device="cuda",batch_size=128):
    plan=training_plan(state,run);seed_all(plan["seed"])
    model=backbone(plan["architecture"]).to(device)
    algorithm=plan["algorithm"]
    def save(epoch,network):
        destination=paths.checkpoint(state,run,epoch)
        if destination.exists():raise FileExistsError("Checkpoint already exists; reference trainer does not overwrite runs")
        destination.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"backbone":{k:v.detach().cpu() for k,v in network.state_dict().items()},
                    "state":state,"run":run,"seed":plan["seed"],"epoch":epoch,"architecture":plan["architecture"],
                    "reference_implementation":True},destination)
    if plan["epochs"]==0:save(0,model);return
    data=Images(paths.inside(plan["dataset"]),algorithm,plan["seed"],plan["retained_correct_fraction"])
    if len(data)!=paths.protocol()["training_images"]:raise ValueError("Reference visual diet must have exactly 1,281,167 images")
    save(0,model)
    loader=DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=algorithm in ["simclr","dino"])
    validation=None;validation_top1=[]
    if algorithm=="supervised":
        validation=Images(paths.inside("data","imagenet","val"),"supervised",plan["seed"])
        if sorted({p.parent.name for p in data.paths})!=sorted({p.parent.name for p in validation.paths}):
            raise ValueError("Training and validation class-directory labels must agree")
        validation.basic=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(227),
                                            transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        validation=DataLoader(validation,batch_size=batch_size,shuffle=False,num_workers=0)
    head=None;teacher=None;center=None
    if algorithm in ["simclr","dino"]:
        # Deterministic fc7: remove dropout and fc8 without altering convolution keys.
        model.classifier=nn.Sequential(*[model.classifier[i] for i in [1,2,4,5]])
        out_dim=128 if algorithm=="simclr" else 65536
        head=nn.Sequential(nn.Linear(4096,4096),nn.ReLU(),nn.Linear(4096,out_dim)).to(device)
        parameters=list(model.parameters())+list(head.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-6 if algorithm=="simclr" else .04)
        if algorithm=="dino":
            teacher=copy.deepcopy(model).eval();teacher_head=copy.deepcopy(head).eval()
            for p in list(teacher.parameters())+list(teacher_head.parameters()):p.requires_grad_(False)
            center=torch.zeros(1,out_dim,device=device)
    else:
        if algorithm=="rotnet":model.classifier[6]=nn.Linear(4096,4).to(device)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=plan["epochs"])
    for epoch in range(1,plan["epochs"]+1):
        model.train()
        for images,y in loader:
            optimizer.zero_grad()
            if algorithm=="simclr":loss=simclr_loss(head(model(images[0].to(device))),head(model(images[1].to(device))))
            elif algorithm=="dino":
                crops=[v.to(device) for v in images]
                students=[head(model(v)) for v in crops]
                with torch.no_grad():teachers=[teacher_head(teacher(v)) for v in crops[:2]]
                loss=dino_loss(students,teachers,center)
            else:
                images=images.to(device);y=y.to(device)
                if algorithm=="rotnet":
                    y=torch.arange(4,device=device).repeat_interleave(len(images))
                    images=torch.cat([torch.rot90(images,k,dims=(-2,-1)) for k in range(4)])
                loss=torch.nn.functional.cross_entropy(model(images),y)
            loss.backward();optimizer.step()
            if algorithm=="dino":
                momentum=.996+(1-.996)*(1-math.cos(math.pi*(epoch-1)/plan["epochs"]))/2
                with torch.no_grad():
                    for student_module,teacher_module in [(model,teacher),(head,teacher_head)]:
                        for p,q in zip(student_module.parameters(),teacher_module.parameters()):q.mul_(momentum).add_(p,alpha=1-momentum)
                    center.mul_(.9).add_(torch.cat(teachers).mean(0,keepdim=True),alpha=.1)
        scheduler.step()
        if validation is not None:
            model.eval();correct=total=0
            with torch.no_grad():
                for images,y in validation:
                    predicted=model(images.to(device)).argmax(1).cpu()
                    correct+=int((predicted==y).sum());total+=len(y)
            validation_top1.append(correct/total)
        if epoch in paths.protocol()["checkpoint_epochs"] or epoch==plan["epochs"]:save(epoch,teacher if algorithm=="dino" else model)
    paths.write_json(paths.checkpoint(state,run,0).parent/"training_manifest.json",{**plan,"validation_top1":validation_top1})
