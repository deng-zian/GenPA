import torch

from genpa.robotwin.depth.model import DepthPolicy


def visual_loss(prediction, velocity, training_weight):
    squared = (prediction.float() - velocity.float()).square()
    return (squared.flatten(1).mean(1) * training_weight.float().reshape(-1)).mean()


class MultiViewPolicy(DepthPolicy):
    def compute_visual_loss(self, sample):
        video = sample["video"]
        batch = video.shape[0]
        reference, reference_ids = self._encode_image_tokens(video[:, :, 0], time_value=10.0)
        target, target_ids = self._encode_image_tokens(video[:, :, 1], time_value=0.0)
        noise = torch.randn_like(target)
        raw_t = self.visual_scheduler.sample_training_t(batch_size=batch, device=target.device, dtype=target.dtype)
        t = raw_t / float(self.visual_scheduler.num_train_timesteps)
        noisy = self.visual_scheduler.add_noise(target, noise, raw_t)
        velocity = self.visual_scheduler.training_target(target, noise, raw_t)
        context = sample["visual_text_hidden_states"].to(device=target.device, dtype=self.torch_dtype)
        context_mask = sample["visual_text_attention_mask"].to(device=target.device, dtype=torch.bool)
        prepared = self._pre_dit_with_reference_timestep(
            noisy_target=noisy, timestep=t, context=context, context_mask=context_mask,
            reference_tokens=reference, reference_ids=reference_ids, target_ids=target_ids,
        )
        attention = self.dit.build_attention_mask(text_mask=prepared["text_mask"],
                                                ref_len=int(prepared["cond_len"]), target_len=int(prepared["target_len"]))
        hidden = self.dit.forward_flux(video_pre=prepared, attention_mask=attention)
        prediction = self.video_expert.post_dit(hidden, prepared)
        loss = visual_loss(prediction, velocity, self.visual_scheduler.training_weight(raw_t))
        return loss, {"loss_visual": loss.detach().item(),
                      f"loss_{self.action_contract['mode']}": loss.detach().item(),
                      "visual_timestep_mean": t.float().mean().item()}
