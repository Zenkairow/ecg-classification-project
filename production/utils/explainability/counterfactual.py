import torch
import torch.nn as nn
import torch.optim as optim

def compute_tv_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Computes the Total Variation (TV) loss for a 1D signal perturbation.
    Encourages physiological smoothness by penalizing high-frequency jumps.
    delta shape: [B, Channels, Length]
    """
    return torch.mean(torch.abs(delta[:, :, 1:] - delta[:, :, :-1]))

def generate_counterfactual(
    model: nn.Module, 
    input_signal: torch.Tensor, 
    target_class_idx: int, 
    epsilon: float = 0.1,
    lambda_l2: float = 0.05,
    lambda_tv: float = 0.1,
    lr: float = 0.01,
    max_steps: int = 100
) -> torch.Tensor:
    """
    Generates a counterfactual ECG signal that alters the model's prediction 
    towards the target_class_idx while minimizing structural perturbation.
    
    Args:
        model: Pre-trained classifier (frozen during generation).
        input_signal: Original ECG tensor [1, 12, L].
        target_class_idx: The integer index of the desired counterfactual class.
        epsilon: Maximum allowed amplitude change per sample (L_inf bound).
        lambda_l2: Weight for the L2 regularization penalty.
        lambda_tv: Weight for the Total Variation (smoothness) penalty.
        lr: Learning rate for the Adam optimizer.
        max_steps: Maximum number of optimization iterations.
        
    Returns:
        counterfactual_signal: The perturbed ECG tensor.
    """
    # 1. Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    device = input_signal.device
    original_signal = input_signal.clone().detach()
    
    # 2. Clone input and enable gradients for the perturbation
    delta = torch.zeros_like(original_signal, requires_grad=True, device=device)
    
    # 3. Setup standard optimizer targeting only the delta tensor
    optimizer = optim.Adam([delta], lr=lr)
    criterion = nn.CrossEntropyLoss()
    target_tensor = torch.tensor([target_class_idx], device=device)
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # Apply perturbation
        modified_signal = original_signal + delta
        
        # Forward pass
        logits = model(modified_signal)
        
        # Calculate losses
        loss_ce = criterion(logits, target_tensor)
        
        # 4. L2 and TV Penalties
        loss_l2 = torch.norm(delta, p=2)
        loss_tv = compute_tv_loss(delta)
        
        # Total Objective
        loss = loss_ce + (lambda_l2 * loss_l2) + (lambda_tv * loss_tv)
        
        loss.backward()
        optimizer.step()
        
        # 5. L_inf projection (Clamp the perturbation to epsilon bounds)
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
            
        # Early Stopping condition
        if step % 5 == 0:
            probs = torch.softmax(logits, dim=1)
            current_pred = torch.argmax(probs, dim=1).item()
            if current_pred == target_class_idx and probs[0, target_class_idx].item() > 0.95:
                break
                
    # Final reconstruction
    with torch.no_grad():
        final_counterfactual = original_signal + delta
        
    return final_counterfactual.detach()
