from __future__ import annotations
import numpy as np # type: ignore
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments # type: ignore
from governed_nlp.modeling.metrics import weighted_kappa, adjacent_accuracy # type: ignore


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_kappa": weighted_kappa(labels, preds),
        "adjacent_acc": adjacent_accuracy(labels, preds),
        "accuracy": float((preds == labels).mean()),
    }


def build_trainer(
    model_name: str,
    num_labels: int,
    train_dataset,
    eval_dataset,
    output_dir: str = "./artifacts",
) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # keep demo fast; increase for real runs
        load_best_model_at_end=True,
        metric_for_best_model="weighted_kappa",
        logging_steps=50,
        report_to="none",
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
