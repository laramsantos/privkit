import torch
import models.seq2seq as seq2seq
import config

def train_model(train_loader):
    """Train the seq2seq trajectory-prediction model and save its weights to model.pth."""
    encoder = seq2seq.Encoder(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,   # e.g., 2+
        p=config.dropout_p              # e.g., 0.2–0.4
    )
    decoder = seq2seq.Decoder(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=config.num_layers,
        p=config.dropout_p
    )
    model = seq2seq.Seq2Seq(encoder, decoder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.L2_penalty)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    teacher = config.teacher_ratio
    for epoch in range(config.epochs):
        model.train()  # dropout active during training
        epoch_loss = 0.0
        for seq_input, seq_target, _, _ in train_loader:
            seq_input = seq_input.to(device)
            seq_target = seq_target.to(device)

            assert torch.isfinite(seq_input).all() and torch.isfinite(seq_target).all(), "NaN/Inf in batch!"

            optimizer.zero_grad()
            output = model(seq_input, seq_target, teacher)
            loss = criterion(output, seq_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        
    return model