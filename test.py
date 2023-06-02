from tensorboardX import SummaryWriter

summary_writer = SummaryWriter(log_dir='')

f = open("nnn.log", "r")
lines = f.readlines()
for i in range(len(lines)):
    l = lines[i].split()
    if len(l) == 5:
        if l[2] == "Training/contrastive_feat_loss":
            summary_writer.add_scalar('Train/contrastive_feat_loss', float(l[3]), l[4])
        if l[2] == "Training/contrastive_out_loss":
            summary_writer.add_scalar('Train/contrastive_out_loss', float(l[3]), l[4])
    if len(l) == 6:
        if l[2] == "Training/Supervised":
            summary_writer.add_scalar('Train/loss_supervised', float(l[4]), l[5])
        if l[2] == "Training/Unsupervised":
            summary_writer.add_scalar('Train/loss_unsupervised', float(l[4]), l[5])

f = open("aaa.log", "r")
lines = f.readlines()
for i in range(len(lines)):
    l = lines[i].split()
    if len(l) == 7:
        summary_writer.add_scalar('Train/contrastive_feat_loss', float(l[3]), l[6])
        summary_writer.add_scalar('Train/contrastive_out_loss', float(l[5]), l[6])
    if len(l) == 9:
        summary_writer.add_scalar('Train/loss_supervised', float(l[4]), l[8])
        summary_writer.add_scalar('Train/loss_unsupervised', float(l[7]), l[8])