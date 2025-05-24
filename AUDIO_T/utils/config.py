class Config:
    # Audio processing parameters
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 2  # in seconds
    N_FFT = 1024
    N_MELS = 128
    HOP_LENGTH = 512
    
    # Training parameters
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    LEARNING_RATE = 5e-5
    MAX_EPOCHS = 2
    
    # Model parameters
    EMBEDDING_DIM = 768
    NUM_HEADS = 2
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.25
    
    # Data paths
    DATA_DIR = '../data'
    
    # Expected dimensions
    EXPECTED_TIME_LENGTH = int((SAMPLE_RATE * MAX_AUDIO_LENGTH) / HOP_LENGTH) + 1
    EXPECTED_WAVEFORM_LENGTH = int(SAMPLE_RATE * MAX_AUDIO_LENGTH)
    EXPECTED_MEL_SPEC_SHAPE = (1, N_MELS, EXPECTED_TIME_LENGTH)
