model = BiLSTM_attn(
    kmer=21,
    hidden_size=256,
    embed_size=[16, 4],
    dropout_rate=0.3,
    num_layer1=2,
    num_layer2=2,
    num_classes=2
)
model.load_state_dict(torch.load("/tmp/model/LSTM_attn_drop3_l2b21_s15_epoch11_accuracy:0.9587.pt"))


kmer = torch.randint(0, 4, [32, 21]).to(torch.int32).cuda()
signal = torch.rand(32,21,19).to(torch.float16).cuda()
model = model.eval().cuda().half()

module = torch.jit.trace(model, (kmer, signal))

module.save("/public3/YHC/model_methylation/trace_scirpt_module_half_acc:0.9587.pt")


model = CTC_encoder(n_hid=512)
model.load_state_dict(torch.load("/tmp/model/CTC_epoch:1_loss:0.322378_model.pt"))
model = model.cuda()
model.eval()

datas = torch.rand(32, 1, 6000).cuda().half()
model.half()
traced_script_module = torch.jit.trace(model, datas)
traced_script_module.save("/public3/YHC/model_merge_ara_oryza_fruitfly/CTC_0922_script_epoch:20_loss:0.267006_model.pt")
