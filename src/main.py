from tv_framework import *

EPOCHS = 10

def start_training(dn:str, use_gpu_n:int):
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model1 = DenseNet201ABENN(baseline, n_classes, False)

    cpf1 = ComparatorFramewok(epochs=EPOCHS, model=model1, use_gpu_n=use_gpu_n, dataset_name=dn, use_augmentation=False)

    def train_step(engine, batch):
        cpf1.model.train()
        cpf1.optimizer.zero_grad()
        x, y = batch[0].to(cpf1.device), batch[1].to(cpf1.device)
        y_pred = cpf1.model(x)
        loss = cpf1.criterion(y_pred, y)

        att = cpf1.model.att.detach()
        att = cp.asarray(att)
        cam_normalized = cp.zeros((att.shape[0], att.shape[2], att.shape[3]))

        for i in range(att.shape[0]):
            s = cp.sum(att[i,0,:,:])
            cam_normalized[i,:,:] = cp.divide(att[i,0,:,:], s)

        # Realizando a média dos batches
        m = cp.mean(cam_normalized, axis=0)

        ce = 10*cp.sum(m*cp.log(m))

        loss = loss - ce.item()
                
        loss.backward()
        cpf1.optimizer.step()

        return loss.item()
            
    def validation_step(engine, batch):
        cpf1.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(cpf1.device), batch[1].to(cpf1.device)
            y_pred = cpf1.model(x)
            return y_pred, y


    cpf1.set_custom_train_step(train_step)
    cpf1.set_custom_val_step(validation_step)
        
    cpf1.procedure("ABN")

   #################################################################################################
    
    model2 = resnet50(True)
    model2.fc = nn.Linear(512 * model2.block.expansion, n_classes)
    cpf2 = ComparatorFramewok(epochs=EPOCHS, model=model2, use_gpu_n=use_gpu_n, dataset_name=dn, use_augmentation=False)

    def train_step2(engine, batch):
        cpf2.model.train()
        cpf2.optimizer.zero_grad()
        x, y = batch[0].to(cpf2.device), batch[1].to(cpf2.device)
        att_outputs, outputs, _ = cpf2.model(x)

        att_loss = cpf2.criterion(att_outputs, y)
        per_loss = cpf2.criterion(outputs, y)
        loss = att_loss + per_loss
                
        loss.backward()
        cpf2.optimizer.step()

        return loss.item()
            
    def validation_step2(engine, batch):
        cpf2.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(cpf2.device), batch[1].to(cpf2.device)
            _, y_pred, _ = cpf2.model(x)
            return y_pred, y

    cpf2.set_custom_train_step(train_step2)
    cpf2.set_custom_val_step(validation_step2)
    cpf2.procedure("RESNET_ABN")


if __name__ == "__main__":
    dts1 = ["CR", "LA", "LG", "NHL", "UCSB"]
    procs = []
    device = False
    last = dts1[0]
    for dn in dts1:
        start_training(dn, 0)