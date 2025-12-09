const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const uploadDir = path.join(__dirname, 'uploads');
fs.mkdirSync(uploadDir, { recursive: true });
const upload = multer({ dest: uploadDir });

const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';
const FEATURE_SCRIPT = path.join(__dirname, '..', 'Smallest_Prototype', 'feature.py');

const parseJsonFromOutput = (output) => {
  const lines = output.trim().split('\n').reverse();
  for (const line of lines) {
    try {
      return JSON.parse(line);
    } catch (err) {
      // keep scanning
    }
  }
  return null;
};

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.post('/api/analyze', upload.single('dataset'), (req, res) => {
  const file = req.file;
  const { targetColumn, context, runTraining } = req.body;

  if (!file) {
    return res.status(400).json({ error: 'Dataset file is required.' });
  }

  const args = [
    FEATURE_SCRIPT,
    '--dataset',
    file.path,
    '--context',
    context || 'Analyze the data and suggest a fitting ML task.',
    '--json-output',
  ];

  if (targetColumn) {
    args.push('--target', targetColumn);
  }

  const shouldTrain = runTraining === true || runTraining === 'true' || runTraining === 'on';
  if (shouldTrain) {
    args.push('--train');
  }

  const child = spawn(PYTHON_BIN, args, { cwd: path.join(__dirname, '..') });

  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  child.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  child.on('error', (err) => {
    return res.status(500).json({ error: 'Failed to start Python process', details: err.message });
  });

  child.on('close', (code) => {
    const parsed = parseJsonFromOutput(stdout);
    if (!parsed) {
      return res.status(500).json({
        error: 'Unable to parse pipeline output',
        exitCode: code,
        stdout,
        stderr,
      });
    }

    const statusCode = parsed.status === 'ok' ? 200 : 400;
    return res.status(statusCode).json({ ...parsed, exitCode: code, logs: stdout, errors: stderr });
  });
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`MLBot server running on port ${PORT}`);
});
