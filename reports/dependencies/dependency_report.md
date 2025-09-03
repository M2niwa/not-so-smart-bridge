# 模块依赖关系报告

## 统计信息

- 总模块数: 32
- 外部依赖数: 39
- 平均依赖数: 0.81

## 外部依赖

- RPi
- adaptive_network
- ai_model
- argparse
- asyncio
- cv2
- datetime
- detector
- feedback_analyzer
- feedback_collector
- feedback_model
- io
- itertools
- json
- layout_model
- layout_optimizer
- logging
- matplotlib
- models
- numpy
- os
- pandas
- pathlib
- performance_monitor
- pickle
- queue
- random
- seaborn
- self_learning_controller
- shutil
- sklearn
- sys
- temperature_model
- temperature_optimizer
- tensorflow
- threading
- time
- tkinter
- typing

## 模块依赖详情

### ai.inference.__init__

**文件路径**: `ai\inference\__init__.py`

**外部依赖**:
- detector

### ai.inference.detector

**文件路径**: `ai\inference\detector.py`

**外部依赖**:
- json
- cv2
- pathlib
- numpy
- logging

### ai.inference.food_layout_inference

**文件路径**: `ai\inference\food_layout_inference.py`

**外部依赖**:
- json
- os
- tensorflow
- typing
- time
- itertools
- numpy
- logging

### ai.self_learning.__init__

**文件路径**: `ai\self_learning\__init__.py`

**外部依赖**:
- self_learning_controller
- adaptive_network
- feedback_collector

### ai.self_learning.adaptive_network

**文件路径**: `ai\self_learning\adaptive_network.py`

**内部依赖**:
- src.core.config
- src.core.food_layout_optimizer
- src.hardware.hardware_controller

**外部依赖**:
- json
- models
- datetime
- typing
- pathlib
- numpy
- logging

### ai.self_learning.example_usage

**文件路径**: `ai\self_learning\example_usage.py`

**内部依赖**:
- src.core.config
- src.ai.self_learning.self_learning_controller

**外部依赖**:
- threading
- time
- datetime
- logging

### ai.self_learning.feedback_analyzer

**文件路径**: `ai\self_learning\feedback_analyzer.py`

**外部依赖**:
- typing
- adaptive_network
- datetime
- logging

### ai.self_learning.feedback_collector

**文件路径**: `ai\self_learning\feedback_collector.py`

**内部依赖**:
- src.core.config
- src.hardware.hardware_controller

**外部依赖**:
- json
- datetime
- tkinter
- typing
- pathlib
- adaptive_network
- logging

### ai.self_learning.layout_optimizer

**文件路径**: `ai\self_learning\layout_optimizer.py`

**内部依赖**:
- src.hardware.hardware_controller

**外部依赖**:
- typing
- adaptive_network
- datetime
- logging

### ai.self_learning.models.__init__

**文件路径**: `ai\self_learning\models\__init__.py`

**外部依赖**:
- feedback_model
- temperature_model
- layout_model

### ai.self_learning.models.feedback_model

**文件路径**: `ai\self_learning\models\feedback_model.py`

**外部依赖**:
- json
- datetime
- tensorflow
- pathlib
- numpy
- pickle
- logging

### ai.self_learning.models.layout_model

**文件路径**: `ai\self_learning\models\layout_model.py`

**外部依赖**:
- json
- tensorflow
- pathlib
- numpy
- pickle
- logging

### ai.self_learning.models.temperature_model

**文件路径**: `ai\self_learning\models\temperature_model.py`

**外部依赖**:
- json
- tensorflow
- pathlib
- numpy
- pickle
- logging

### ai.self_learning.performance_monitor

**文件路径**: `ai\self_learning\performance_monitor.py`

**外部依赖**:
- json
- datetime
- typing
- pathlib
- logging

### ai.self_learning.self_learning_controller

**文件路径**: `ai\self_learning\self_learning_controller.py`

**内部依赖**:
- src.core.config
- src.hardware.temperature_optimizer
- src.core.food_layout_optimizer
- src.hardware.hardware_controller

**外部依赖**:
- json
- temperature_optimizer
- threading
- datetime
- performance_monitor
- typing
- time
- feedback_collector
- feedback_analyzer
- layout_optimizer
- pathlib
- adaptive_network
- logging

### ai.self_learning.temperature_optimizer

**文件路径**: `ai\self_learning\temperature_optimizer.py`

**内部依赖**:
- src.hardware.hardware_controller

**外部依赖**:
- typing
- adaptive_network
- datetime
- logging

### ai.training.data_preprocessor

**文件路径**: `ai\training\data_preprocessor.py`

**外部依赖**:
- pandas
- json
- tensorflow
- cv2
- sklearn
- pathlib
- numpy
- pickle
- logging

### ai.training.example_usage

**文件路径**: `ai\training\example_usage.py`

**外部依赖**:
- os
- ai_model
- pathlib
- sys
- numpy
- logging

### ai.training.food_classifier_trainer

**文件路径**: `ai\training\food_classifier_trainer.py`

**外部依赖**:
- json
- matplotlib
- sys
- io
- tensorflow
- pathlib
- numpy
- logging

### ai.training.food_layout_model

**文件路径**: `ai\training\food_layout_model.py`

**外部依赖**:
- json
- os
- tensorflow
- typing
- numpy
- logging

### ai.training.model_evaluator

**文件路径**: `ai\training\model_evaluator.py`

**外部依赖**:
- json
- pandas
- seaborn
- matplotlib
- tensorflow
- sklearn
- time
- pathlib
- numpy
- logging

### ai.training.model_trainer

**文件路径**: `ai\training\model_trainer.py`

**外部依赖**:
- pandas
- json
- tensorflow
- sklearn
- pathlib
- numpy
- pickle
- logging

### ai.training.prepare_training_data

**文件路径**: `ai\training\prepare_training_data.py`

**外部依赖**:
- json
- pandas
- os
- datetime
- shutil
- pathlib
- cv2
- sklearn
- argparse
- sys
- numpy
- logging

### ai.training.train_food_classifier

**文件路径**: `ai\training\train_food_classifier.py`

**内部依赖**:
- src.ai.training.model_evaluator
- src.ai.training.model_trainer
- src.ai.training.data_preprocessor
- src.ai.training.food_classifier_trainer

**外部依赖**:
- json
- os
- pathlib
- argparse
- sys
- logging

### core.async_image_processor

**文件路径**: `core\async_image_processor.py`

**内部依赖**:
- src.ai.inference.detector
- src.hardware.hardware_controller

**外部依赖**:
- threading
- datetime
- sys
- queue
- time
- pathlib
- asyncio
- logging

### core.config

**文件路径**: `core\config.py`

### core.firmware_interface

**文件路径**: `core\firmware_interface.py`

**外部依赖**:
- random
- cv2
- RPi

### core.food_layout_optimizer

**文件路径**: `core\food_layout_optimizer.py`

**内部依赖**:
- src.ai.inference.detector
- src.hardware.hardware_controller

**外部依赖**:
- random
- typing
- itertools
- logging

### core.main

**文件路径**: `core\main.py`

**内部依赖**:
- src.ai.inference.detector
- src.hardware.hardware_controller

**外部依赖**:
- threading
- json
- os
- datetime
- shutil
- pathlib
- time
- sys
- logging

### hardware.hardware_controller

**文件路径**: `hardware\hardware_controller.py`

**内部依赖**:
- src.core.config

**外部依赖**:
- random
- temperature_optimizer
- time
- logging

### hardware.temperature_optimizer

**文件路径**: `hardware\temperature_optimizer.py`

**内部依赖**:
- src.core

### main

**文件路径**: `main.py`

**内部依赖**:
- src.core.main

**外部依赖**:
- sys
- pathlib

