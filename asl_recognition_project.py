

# =============================================================================
# SECTION 1 – DATASET DOWNLOAD
# =============================================================================

def download_dataset() -> str:
    """
    Downloads the ASL dataset from Kaggle using kagglehub and returns the
    local path where the files are stored.

    Returns
    -------
    str
        Root directory containing 'ASL_Dataset/Train' and 'ASL_Dataset/Test'.
    """
    import kagglehub
    path = kagglehub.dataset_download("kapillondhe/american-sign-language")
    print(f"Dataset downloaded to: {path}")
    return path


# =============================================================================
# SECTION 2 – PREPROCESSING UTILITIES
# =============================================================================

def load_images_grayscale(
    folder_path: str,
    img_size: int = 64,
    max_per_class: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads, preprocesses, and returns images as NumPy arrays.

    Preprocessing pipeline per image:
      1. Grayscale conversion (BGR → GRAY)
      2. Otsu's thresholding to create a binary hand-region mask
      3. Bitwise-AND masking to remove background pixels
      4. Resize to (img_size × img_size)
      5. Pixel normalisation to [0, 1]

    Parameters
    ----------
    folder_path   : Root directory with one sub-folder per class.
    img_size      : Target height/width in pixels.
    max_per_class : Maximum images loaded per class (keeps RAM manageable).

    Returns
    -------
    images : float32 array of shape (N, img_size, img_size)
    labels : str array of shape (N,) — raw class-folder names
    """
    images, labels = [], []

    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for img_name in os.listdir(class_path):
            if count >= max_per_class:
                break

            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None:
                continue  # skip corrupted / unreadable files

            # Step 1: convert to single-channel greyscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Step 2: Otsu's binary mask (inverted — hand is brighter on dark bg)
            _, mask = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Step 3: suppress background
            img = cv2.bitwise_and(img, img, mask=mask)

            # Steps 4 & 5: resize + normalise
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0

            images.append(img)
            labels.append(class_name)
            count += 1

    return np.array(images, dtype="float32"), np.array(labels)


def load_images_rgb(
    folder_path: str,
    img_size: int = 96,
    max_per_class: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads images as RGB float32 arrays for transfer-learning models (VGG16 /
    MobileNet) which expect 3-channel inputs.

    Parameters
    ----------
    folder_path   : Root directory with one sub-folder per class.
    img_size      : Target height/width (96 recommended for pretrained models).
    max_per_class : Maximum images per class.

    Returns
    -------
    images : float32 array of shape (N, img_size, img_size, 3)
    labels : str array of shape (N,)
    """
    images, labels = [], []

    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for img_name in os.listdir(class_path):
            if count >= max_per_class:
                break

            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # OpenCV reads BGR
            img = cv2.resize(img, (img_size, img_size))
            images.append(img.astype("float32") / 255.0)
            labels.append(class_name)
            count += 1

    return np.array(images, dtype="float32"), np.array(labels)


# =============================================================================
# SECTION 3 – MACHINE LEARNING MODELS (SVM & KNN)
# =============================================================================

def run_svm(X_train_flat, y_train_enc, X_test_flat, y_test_enc, label_encoder):
    """
    Trains a linear-kernel SVM on flattened pixel features and prints the
    accuracy + full classification report.

    Why linear kernel?
    ------------------
    High-dimensional pixel feature vectors (64×64 = 4096 dims) are often
    linearly separable — especially after Otsu masking reduces noise.
    A linear kernel also avoids the quadratic memory cost of RBF kernels.

    Parameters
    ----------
    X_train_flat  : (N_train, 4096) flat pixel features.
    y_train_enc   : Integer-encoded train labels.
    X_test_flat   : (N_test, 4096) flat pixel features.
    y_test_enc    : Integer-encoded test labels.
    label_encoder : Fitted LabelEncoder (used for class names in the report).

    Returns
    -------
    svm_model : Trained SVC instance.
    svm_pred  : Predicted labels for the test set.
    """
    print("\n" + "="*60)
    print("TRAINING: Support Vector Machine (Linear Kernel)")
    print("="*60)

    svm_model = SVC(kernel="linear", random_state=SEED)
    svm_model.fit(X_train_flat, y_train_enc)

    svm_pred = svm_model.predict(X_test_flat)

    print(f"\nSVM Test Accuracy : {accuracy_score(y_test_enc, svm_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_enc, svm_pred, target_names=label_encoder.classes_))

    return svm_model, svm_pred


def run_knn(X_train_flat, y_train_enc, X_test_flat, y_test_enc, label_encoder, k: int = 3):
    """
    Trains a K-Nearest Neighbours classifier on flattened pixel features.

    k=3 is chosen as a balance between bias (underfitting at k=1) and variance
    (over-smoothing at large k).  For ASL gestures, neighbouring samples in
    pixel space tend to correspond to the same letter sign.

    Parameters
    ----------
    X_train_flat  : (N_train, 4096) flat pixel features.
    y_train_enc   : Integer-encoded train labels.
    X_test_flat   : (N_test, 4096) flat pixel features.
    y_test_enc    : Integer-encoded test labels.
    label_encoder : Fitted LabelEncoder.
    k             : Number of nearest neighbours.

    Returns
    -------
    knn_model : Trained KNeighborsClassifier instance.
    knn_pred  : Predicted labels for the test set.
    """
    print("\n" + "="*60)
    print(f"TRAINING: K-Nearest Neighbours (k={k})")
    print("="*60)

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_flat, y_train_enc)

    knn_pred = knn_model.predict(X_test_flat)

    print(f"\nKNN Test Accuracy : {accuracy_score(y_test_enc, knn_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_enc, knn_pred, target_names=label_encoder.classes_))

    return knn_model, knn_pred


# =============================================================================
# SECTION 4 – CUSTOM CNN (DEEP LEARNING)
# =============================================================================

def build_custom_cnn(num_classes: int = 28, input_shape: tuple = (64, 64, 1)) -> tf.keras.Model:
    """
    Constructs and compiles the custom CNN architecture.

    Architecture overview
    ---------------------
    Block 1 : Conv(32) → BN → Conv(32) → MaxPool → Dropout(0.25)
    Block 2 : Conv(64) → BN → Conv(64) → MaxPool → Dropout(0.30)
    Block 3 : Conv(128) → BN          → MaxPool → Dropout(0.35)
    Head    : Flatten → Dense(256, ReLU) → Dropout(0.40) → Dense(28, Softmax)

    Design rationale
    ----------------
    * Paired convolutions before pooling (similar to VGG block style) capture
      richer local features without quadratic parameter growth.
    * Batch normalisation stabilises training and acts as implicit regularisation.
    * Increasing dropout rates with depth combat overfitting in deeper layers.
    * Adam (lr=1e-4) with ReduceLROnPlateau provides adaptive learning.

    Parameters
    ----------
    num_classes  : Number of output classes (28 for ASL).
    input_shape  : (height, width, channels).

    Returns
    -------
    Compiled tf.keras.Model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # ── Block 1 ──────────────────────────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # ── Block 3 ──────────────────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        # ── Classifier head ──────────────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Custom_CNN")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_custom_cnn(X_train_dl, y_train_dl, X_test_dl, y_test_cat,
                     num_classes: int = 28, epochs: int = 20):
    """
    Trains the custom CNN with data augmentation, early stopping, and LR decay.

    Augmentation parameters (conservative, preserving sign semantics)
    -----------------------------------------------------------------
    rotation_range=10    : Slight tilt — real-world hand angle variation.
    zoom_range=0.1       : Small scale variation — camera distance differences.
    width/height_shift   : Translational invariance within frame.
    horizontal_flip=False: DISABLED — left/right handedness matters in ASL.

    Training callbacks
    ------------------
    EarlyStopping  : Stops training if val_loss fails to improve for 5 epochs,
                     then restores best weights.
    ReduceLROnPlateau: Halves LR when val_loss plateaus for 2 epochs (min 1e-6).

    Parameters
    ----------
    X_train_dl  : (N, 64, 64, 1) normalised grayscale training images.
    y_train_dl  : Integer-encoded labels (used for stratified split).
    X_test_dl   : (N_test, 64, 64, 1) test images.
    y_test_cat  : One-hot encoded test labels.
    num_classes : 28 for ASL.
    epochs      : Maximum training epochs (early stopping typically fires first).

    Returns
    -------
    cnn_model : Trained model.
    history   : Keras History object (loss/accuracy curves).
    test_acc  : Final test accuracy (float).
    """
    # ── Stratified train/val split (85% / 15%) ───────────────────────────────
    y_train_cat = to_categorical(y_train_dl, num_classes=num_classes)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_dl, y_train_cat,
        test_size=0.15, random_state=SEED, stratify=y_train_dl
    )

    # ── Data augmentation ────────────────────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        fill_mode="nearest"
    )
    datagen.fit(X_tr)

    # ── Build & train ────────────────────────────────────────────────────────
    cnn_model = build_custom_cnn(num_classes=num_classes)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    print("\n" + "="*60)
    print("TRAINING: Custom CNN")
    print("="*60)

    history = cnn_model.fit(
        datagen.flow(X_tr, y_tr, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    test_loss, test_acc = cnn_model.evaluate(X_test_dl, y_test_cat, verbose=0)
    print(f"\nCNN Test Loss     : {test_loss:.4f}")
    print(f"CNN Test Accuracy : {test_acc:.4f}")

    return cnn_model, history, test_acc


# =============================================================================
# SECTION 5 – TRANSFER LEARNING (VGG16 & MobileNet)
# =============================================================================

def build_transfer_model(
    base_arch: str = "vgg16",
    num_classes: int = 28,
    input_shape: tuple = (96, 96, 3)
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds a transfer-learning model by attaching a custom classification head
    to an ImageNet-pretrained base (VGG16 or MobileNet) with frozen weights.

    Transfer learning strategy
    --------------------------
    Phase 1 (this function): Freeze all base layers → train only the head.
              Fast convergence; base features act as fixed feature extractors.
    Phase 2 (fine_tune function): Unfreeze top N layers → re-train at low LR.
              Allows domain-specific adaptation without catastrophic forgetting.

    Parameters
    ----------
    base_arch   : "vgg16" or "mobilenet".
    num_classes : 28 for ASL.
    input_shape : (H, W, 3) — must be ≥ 32×32 for VGG16, ≥ 32×32 for MobileNet.

    Returns
    -------
    model      : Full compiled Keras Model.
    base_model : Reference to the pretrained base (used in fine-tuning).
    """
    if base_arch == "vgg16":
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif base_arch == "mobilenet":
        base_model = MobileNet(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown architecture: {base_arch}")

    # Freeze all pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Attach classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)           # replaces Flatten (more robust)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output,
                  name=f"{base_arch}_transfer")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def fine_tune_vgg16(model: tf.keras.Model, base_model: tf.keras.Model,
                    X_train_aug, y_train_aug, X_val_aug, y_val_aug,
                    datagen: ImageDataGenerator, unfreeze_last: int = 4,
                    epochs: int = 2):
    """
    Fine-tunes VGG16 by unfreezing its top `unfreeze_last` convolutional layers
    and re-training at a very low learning rate.

    Why a low LR (1e-5)?
    --------------------
    Pretrained weights encode general visual features (edges, textures, shapes).
    A high LR would overwrite these features — known as catastrophic forgetting.
    lr=1e-5 nudges weights gently toward ASL-specific patterns.

    Parameters
    ----------
    model          : Full VGG16 transfer model from build_transfer_model().
    base_model     : Reference to the VGG16 base.
    X_train_aug    : Training images (float32, normalised).
    y_train_aug    : Integer class labels.
    X_val_aug      : Validation images.
    y_val_aug      : Validation labels.
    datagen        : Pre-fit ImageDataGenerator.
    unfreeze_last  : Number of base layers to unfreeze (from the top).
    epochs         : Fine-tuning epochs.

    Returns
    -------
    history_fine : Keras History object for the fine-tuning phase.
    """
    # Unfreeze all, then re-freeze everything except top N layers
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nFine-tuning: unfroze last {unfreeze_last} VGG16 layers (lr=1e-5)")

    history_fine = model.fit(
        datagen.flow(X_train_aug, y_train_aug, batch_size=32),
        epochs=epochs,
        validation_data=(X_val_aug, y_val_aug),
        verbose=1
    )

    return history_fine


def run_transfer_models(X_train_pre, y_train_idx, X_test_pre, y_test_idx,
                        label_encoder, num_classes: int = 28):
    """
    Orchestrates the full transfer-learning pipeline:
      1. Split train → train/val (80/20, stratified)
      2. Build augmentation generator
      3. Train VGG16 (phase 1) → fine-tune (phase 2)
      4. Train MobileNet (phase 1 only)
      5. Evaluate and print results for both models

    Parameters
    ----------
    X_train_pre   : (N, 96, 96, 3) RGB float32 training images.
    y_train_idx   : Integer-encoded train labels.
    X_test_pre    : (N_test, 96, 96, 3) RGB float32 test images.
    y_test_idx    : Integer-encoded test labels.
    label_encoder : Fitted LabelEncoder.
    num_classes   : 28.

    Returns
    -------
    model_vgg    : Trained VGG16 model.
    model_mobile : Trained MobileNet model.
    history_vgg  : (phase1_history, fine_tune_history)
    history_mob  : MobileNet history.
    """
    # ── 1. Train / val split ─────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_pre, y_train_idx,
        test_size=0.2, random_state=SEED, stratify=y_train_idx
    )

    # ── 2. Augmentation ──────────────────────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    )
    datagen.fit(X_tr)

    # ── 3. VGG16 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: VGG16 — Phase 1 (Transfer Learning, frozen base)")
    print("="*60)

    model_vgg, base_vgg = build_transfer_model("vgg16", num_classes)
    history_vgg_p1 = model_vgg.fit(
        datagen.flow(X_tr, y_tr, batch_size=32),
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Phase 2: fine-tuning
    print("\n" + "="*60)
    print("TRAINING: VGG16 — Phase 2 (Fine-tuning, last 4 layers)")
    print("="*60)

    history_vgg_ft = fine_tune_vgg16(
        model_vgg, base_vgg, X_tr, y_tr, X_val, y_val, datagen
    )

    # Evaluate VGG16
    y_pred_vgg = np.argmax(model_vgg.predict(X_test_pre, verbose=0), axis=1)
    print(f"\nVGG16 Test Accuracy : {accuracy_score(y_test_idx, y_pred_vgg):.4f}")
    print(classification_report(y_test_idx, y_pred_vgg, target_names=label_encoder.classes_))

    # ── 4. MobileNet ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: MobileNet (Transfer Learning, frozen base)")
    print("="*60)

    model_mobile, _ = build_transfer_model("mobilenet", num_classes)
    history_mob = model_mobile.fit(
        datagen.flow(X_tr, y_tr, batch_size=32),
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate MobileNet
    y_pred_mobile = np.argmax(model_mobile.predict(X_test_pre, verbose=0), axis=1)
    print(f"\nMobileNet Test Accuracy : {accuracy_score(y_test_idx, y_pred_mobile):.4f}")
    print(classification_report(y_test_idx, y_pred_mobile, target_names=label_encoder.classes_))

    return (model_vgg, model_mobile,
            (history_vgg_p1, history_vgg_ft), history_mob,
            y_pred_vgg, y_pred_mobile)


# =============================================================================
# SECTION 6 – VISUALISATION UTILITIES
# =============================================================================

def plot_sample_images(X_dl, y_enc, label_encoder, n: int = 5):
    """Displays a grid of n sample images with their class labels."""
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_dl[i].reshape(64, 64), cmap="gray")
        ax.set_title(f"Label: {label_encoder.classes_[y_enc[i]]}")
        ax.axis("off")
    plt.suptitle("Sample Preprocessed Training Images")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title_prefix: str = "Model"):
    """
    Plots training vs. validation accuracy and loss curves from a Keras History.
    """
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc,     label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title(f"{title_prefix} – Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(loss,     label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_vgg16_combined_curves(history_p1, history_ft):
    """
    Plots combined VGG16 Phase-1 + Fine-tuning curves with a dashed separator.
    """
    acc      = history_p1.history["accuracy"]      + history_ft.history["accuracy"]
    val_acc  = history_p1.history["val_accuracy"]  + history_ft.history["val_accuracy"]
    loss     = history_p1.history["loss"]           + history_ft.history["loss"]
    val_loss = history_p1.history["val_loss"]       + history_ft.history["val_loss"]
    ep1      = len(history_p1.history["accuracy"])  # index where fine-tuning starts

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.axvline(ep1 - 1, color="red", linestyle="--", label="Fine-tune start")
    plt.title("VGG16 – Accuracy (Transfer → Fine-tune)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.axvline(ep1 - 1, color="red", linestyle="--", label="Fine-tune start")
    plt.title("VGG16 – Loss (Transfer → Fine-tune)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_encoder, title: str, cmap: str = "Blues"):
    """Plots a colour-mapped confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap=cmap
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_sample_predictions(cnn_model, X_test_dl, y_test_encoded, label_encoder, n: int = 5):
    """Shows n random test images with their true and predicted labels."""
    indices = random.sample(range(len(X_test_dl)), n)
    plt.figure(figsize=(18, 5))
    for i, idx in enumerate(indices):
        img        = X_test_dl[idx]
        true_label = label_encoder.classes_[y_test_encoded[idx]]
        probs      = cnn_model.predict(img.reshape(1, 64, 64, 1), verbose=0)
        pred_label = label_encoder.classes_[np.argmax(probs)]

        plt.subplot(1, n, i + 1)
        plt.imshow(img.reshape(64, 64), cmap="gray")
        color = "green" if pred_label == true_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")

    plt.suptitle("CNN – Sample Predictions (green=correct, red=wrong)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: dict):
    """
    Bar chart comparing test accuracy across all five models.

    Parameters
    ----------
    results : dict mapping model name → accuracy (float between 0 and 1).
              Example: {"SVM": 0.85, "KNN": 0.72, "CNN": 0.91, ...}
    """
    import pandas as pd

    df = pd.DataFrame({
        "Model"    : list(results.keys()),
        "Accuracy" : list(results.values()),
        "Type"     : ["Classical ML", "Classical ML",
                      "Deep Learning", "Transfer Learning", "Transfer Learning"]
    }).sort_values("Accuracy", ascending=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── All models bar chart ──────────────────────────────────────────────────
    ax = sns.barplot(data=df, x="Accuracy", y="Model", palette="mako", ax=axes[0])
    axes[0].set_title("All Models – Test Accuracy", fontweight="bold")
    axes[0].set_xlim(0.6, 1.02)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=6)

    # ── Average by model type ─────────────────────────────────────────────────
    type_avg = df.groupby("Type")["Accuracy"].mean().sort_values()
    ax2 = sns.barplot(x=type_avg.values, y=type_avg.index, palette="coolwarm", ax=axes[1])
    axes[1].set_title("Average Accuracy by Model Type", fontweight="bold")
    for i, v in enumerate(type_avg.values):
        axes[1].text(v + 0.002, i, f"{v:.4f}", va="center")

    sns.despine()
    plt.tight_layout()
    plt.show()

    print("\nFinal Accuracy Summary:")
    print(df[["Model", "Type", "Accuracy"]].sort_values("Accuracy", ascending=False).to_string(index=False))


# =============================================================================
# SECTION 7 – MAIN PIPELINE
# =============================================================================

def main():
    """
    End-to-end ASL recognition pipeline.

    Execution order
    ---------------
    1.  Download dataset
    2.  Load & preprocess grayscale images (64×64) for ML + CNN
    3.  Load & preprocess RGB images (96×96) for transfer learning
    4.  Encode labels, shuffle training data
    5.  Train SVM and KNN
    6.  Train custom CNN (with augmentation, callbacks)
    7.  Train VGG16 (transfer + fine-tuning) and MobileNet
    8.  Visualise all results and produce final comparison chart
    """

    # ── Step 1: Download ──────────────────────────────────────────────────────
    dataset_root = download_dataset()
    train_path   = os.path.join(dataset_root, "ASL_Dataset", "Train")
    test_path    = os.path.join(dataset_root, "ASL_Dataset", "Test")

    # ── Step 2: Grayscale data (ML + CNN) ────────────────────────────────────
    print("\nLoading grayscale images (64×64) ...")
    X_train_gray, y_train_raw = load_images_grayscale(train_path, img_size=64, max_per_class=400)
    X_test_gray,  y_test_raw  = load_images_grayscale(test_path,  img_size=64, max_per_class=500)

    # ── Label encoding ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc  = le.transform(y_test_raw)
    print(f"Classes ({len(le.classes_)}): {le.classes_}")

    # Shuffle training data
    X_train_gray, y_train_enc = shuffle(X_train_gray, y_train_enc, random_state=SEED)

    # ── Feature matrices ──────────────────────────────────────────────────────
    X_train_flat = X_train_gray.reshape(len(X_train_gray), -1)   # (N, 4096) for ML
    X_test_flat  = X_test_gray.reshape(len(X_test_gray),  -1)

    X_train_dl   = X_train_gray.reshape(-1, 64, 64, 1)           # (N, 64, 64, 1) for CNN
    X_test_dl    = X_test_gray.reshape(-1, 64, 64, 1)

    print(f"Grayscale shapes → Train: {X_train_gray.shape}, Test: {X_test_gray.shape}")

    # Visualise sample images
    plot_sample_images(X_train_dl, y_train_enc, le)

    # ── Step 3: RGB data (Transfer Learning) ─────────────────────────────────
    print("\nLoading RGB images (96×96) for transfer learning ...")
    X_train_rgb, y_train_rgb_raw = load_images_rgb(train_path, img_size=96, max_per_class=300)
    X_test_rgb,  y_test_rgb_raw  = load_images_rgb(test_path,  img_size=96, max_per_class=500)

    y_train_rgb_enc = le.transform(y_train_rgb_raw)
    y_test_rgb_enc  = le.transform(y_test_rgb_raw)
    X_train_rgb, y_train_rgb_enc = shuffle(X_train_rgb, y_train_rgb_enc, random_state=SEED)

    print(f"RGB shapes → Train: {X_train_rgb.shape}, Test: {X_test_rgb.shape}")

    # ── Step 4: SVM ───────────────────────────────────────────────────────────
    svm_model, svm_pred = run_svm(X_train_flat, y_train_enc, X_test_flat, y_test_enc, le)
    svm_acc = accuracy_score(y_test_enc, svm_pred)

    # ── Step 5: KNN ───────────────────────────────────────────────────────────
    knn_model, knn_pred = run_knn(X_train_flat, y_train_enc, X_test_flat, y_test_enc, le)
    knn_acc = accuracy_score(y_test_enc, knn_pred)

    # ── Step 6: Custom CNN ───────────────────────────────────────────────────
    y_test_cat = to_categorical(y_test_enc, num_classes=28)
    cnn_model, cnn_history, cnn_acc = train_custom_cnn(
        X_train_dl, y_train_enc, X_test_dl, y_test_cat
    )

    # CNN visualisations
    plot_training_curves(cnn_history, "Custom CNN")
    plot_sample_predictions(cnn_model, X_test_dl, y_test_enc, le)

    # CNN classification report
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_dl, verbose=0), axis=1)
    print("\nCNN – Classification Report:")
    print(classification_report(y_test_enc, y_pred_cnn, target_names=le.classes_))

    # ── Step 7: Transfer Learning ────────────────────────────────────────────
    (model_vgg, model_mobile,
     (hist_vgg_p1, hist_vgg_ft), hist_mob,
     y_pred_vgg, y_pred_mobile) = run_transfer_models(
        X_train_rgb, y_train_rgb_enc, X_test_rgb, y_test_rgb_enc, le
    )

    vgg_acc    = accuracy_score(y_test_rgb_enc, y_pred_vgg)
    mobile_acc = accuracy_score(y_test_rgb_enc, y_pred_mobile)

    # Transfer model visualisations
    plot_vgg16_combined_curves(hist_vgg_p1, hist_vgg_ft)
    plot_training_curves(hist_mob, "MobileNet")
    plot_confusion_matrix(y_test_rgb_enc, y_pred_vgg,    le, "VGG16 – Confusion Matrix",    "Purples")
    plot_confusion_matrix(y_test_rgb_enc, y_pred_mobile, le, "MobileNet – Confusion Matrix", "Greens")

    # ── Step 8: Final Comparison ──────────────────────────────────────────────
    results = {
        "SVM (Linear)"        : svm_acc,
        "KNN (k=3)"           : knn_acc,
        "Custom CNN"          : cnn_acc,
        "MobileNet"           : mobile_acc,
        "VGG16 (Fine-tuned)"  : vgg_acc,
    }

    plot_model_comparison(results)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
