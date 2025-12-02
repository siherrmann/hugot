//go:build ORT || ALL

package pipelines

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

// ImageToTextPipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_to_text.py
// It takes images (as file paths or image.Image) and returns generated text captions.
type ImageToTextPipeline struct {
	*pipelineBackends.BasePipeline
	EncoderModel       *pipelineBackends.Model // Vision encoder model
	DecoderModel       *pipelineBackends.Model // Text decoder model (set to Model in BasePipeline)
	MaxNewTokens       int
	Prompt             string
	preprocessSteps    []util.PreprocessStep
	normalizationSteps []util.NormalizationStep
	format             string
}

type ImageToTextResult struct {
	GeneratedText string
}

type ImageToTextOutput struct {
	Results []ImageToTextResult // one result per image in batch
}

func (o *ImageToTextOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = any(result)
	}
	return out
}

func WithImageToTextPreprocessSteps(steps ...util.PreprocessStep) pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(p *ImageToTextPipeline) error {
		p.preprocessSteps = append(p.preprocessSteps, steps...)
		return nil
	}
}

func WithImageToTextNormalizationSteps(steps ...util.NormalizationStep) pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(p *ImageToTextPipeline) error {
		p.normalizationSteps = append(p.normalizationSteps, steps...)
		return nil
	}
}

func WithImageToTextNHWCFormat() pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(pipeline *ImageToTextPipeline) error {
		pipeline.format = "NHWC"
		return nil
	}
}

func WithImageToTextNCHWFormat() pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(pipeline *ImageToTextPipeline) error {
		pipeline.format = "NCHW"
		return nil
	}
}

// WithMaxNewTokens sets the maximum number of tokens to generate.
func WithMaxNewTokens(maxTokens int) pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(pipeline *ImageToTextPipeline) error {
		pipeline.MaxNewTokens = maxTokens
		return nil
	}
}

// WithPrompt sets a conditional prompt for text generation.
func WithPrompt(prompt string) pipelineBackends.PipelineOption[*ImageToTextPipeline] {
	return func(pipeline *ImageToTextPipeline) error {
		pipeline.Prompt = prompt
		return nil
	}
}

func detectImageToTextInputFormat(model *pipelineBackends.Model) (string, error) {
	inputInfo := model.InputsMeta
	if len(inputInfo) == 0 {
		return "", fmt.Errorf("no inputs found in model")
	}

	// Find the image input (usually the first input or one with 4 dimensions)
	for _, input := range inputInfo {
		shape := input.Dimensions
		if len(shape) == 4 {
			// Found a 4D input, infer format from shape if possible
			// Check if any dimension is 3 (channels)
			if len(shape) > 1 && shape[1] == 3 && (len(shape) < 4 || shape[3] != 3) {
				return "NCHW", nil
			} else if len(shape) > 3 && shape[3] == 3 {
				return "NHWC", nil
			}
			// If shape is dynamic (e.g., [-1, -1, -1, -1]), default to NCHW for vision transformers
			if shape[0] == -1 || shape[1] == -1 {
				return "NCHW", nil
			}
			return "", fmt.Errorf("unable to infer format from shape %v", shape)
		}
	}

	return "", fmt.Errorf("no 4D image input found in model")
}

// NewImageToTextPipeline initializes an image-to-text pipeline.
func NewImageToTextPipeline(config pipelineBackends.PipelineConfig[*ImageToTextPipeline], s *options.Options, model *pipelineBackends.Model) (*ImageToTextPipeline, error) {
	// For vision-encoder-decoder models, we need to load both encoder and decoder
	// The model passed in is the decoder, and we need to load the encoder separately
	var encoderModel *pipelineBackends.Model
	var err error
	
	// Try to load encoder model (use quantized version for smaller memory footprint)
	encoderModel, err = pipelineBackends.LoadModel(config.ModelPath, "onnx/encoder_model_quantized.onnx", s)
	if err != nil {
		return nil, fmt.Errorf("failed to load encoder model: %w", err)
	}

	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &ImageToTextPipeline{
		BasePipeline: defaultPipeline,
		EncoderModel: encoderModel,
		DecoderModel: model,
		MaxNewTokens: 256, // default max_new_tokens=256
	}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}

	if pipeline.format == "" {
		detectedFormat, err := detectImageToTextInputFormat(encoderModel)
		if err != nil {
			return nil, err
		}
		pipeline.format = detectedFormat
	}

	// validate pipeline
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATIONS

func (p *ImageToTextPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *ImageToTextPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *ImageToTextPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *ImageToTextPipeline) Validate() error {
	var validationErrors []error

	// Check for tokenizer (required for decoding generated tokens)
	if p.DecoderModel.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("image-to-text pipeline requires a tokenizer"))
	}

	// Check for image input (4D tensor) in the ENCODER model
	hasImageInput := false
	for _, input := range p.EncoderModel.InputsMeta {
		dims := []int64(input.Dimensions)
		if len(dims) == 4 {
			hasImageInput = true
			break
		}
	}

	if !hasImageInput {
		validationErrors = append(validationErrors, fmt.Errorf("no image input found in encoder: expected at least one 4D input tensor"))
	}

	// Check for text output capabilities in decoder
	if len(p.DecoderModel.OutputsMeta) == 0 {
		validationErrors = append(validationErrors, fmt.Errorf("no outputs found in decoder model"))
	}

	return errors.Join(validationErrors...)
}

// Preprocess decodes images from file paths or image.Image and creates input tensors.
func (p *ImageToTextPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []image.Image) error {
	preprocessed, err := p.preprocessImages(inputs)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return pipelineBackends.CreateImageTensors(batch, preprocessed, p.Runtime)
}

func (p *ImageToTextPipeline) preprocessImages(images []image.Image) ([][][][]float32, error) {
	batchSize := len(images)
	out := make([][][][]float32, batchSize)

	for i, img := range images {
		processed := img
		for _, step := range p.preprocessSteps {
			var err error
			processed, err = step.Apply(processed)
			if err != nil {
				return nil, fmt.Errorf("failed to apply preprocessing step: %w", err)
			}
		}

		hh := processed.Bounds().Dy()
		ww := processed.Bounds().Dx()
		c := 3

		switch strings.ToUpper(p.format) {
		case "NHWC":
			// Height × Width × Channels
			tensor := make([][][]float32, hh)
			for y := range hh {
				tensor[y] = make([][]float32, ww)
				for x := range ww {
					tensor[y][x] = make([]float32, c)
				}
			}

			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf := float32(r >> 8)
					gf := float32(g >> 8)
					bf := float32(b >> 8)
					for _, step := range p.normalizationSteps {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[y][x][0] = rf
					tensor[y][x][1] = gf
					tensor[y][x][2] = bf
				}
			}
			out[i] = tensor
		case "NCHW":
			// Channels × Height × Width
			tensor := make([][][]float32, c)
			for ch := range c {
				tensor[ch] = make([][]float32, hh)
				for y := range hh {
					tensor[ch][y] = make([]float32, ww)
				}
			}

			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf := float32(r >> 8)
					gf := float32(g >> 8)
					bf := float32(b >> 8)
					for _, step := range p.normalizationSteps {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[0][y][x] = rf
					tensor[1][y][x] = gf
					tensor[2][y][x] = bf
				}
			}
			out[i] = tensor
		default:
			return nil, fmt.Errorf("unsupported format: %s", p.format)
		}
	}
	return out, nil
}

// Forward runs inference with proper encoder-decoder architecture.
// Supports both single-stage models and multi-stage encoder-decoder models.
func (p *ImageToTextPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	batchSize := batch.Size
	batchSize64 := int64(batchSize)
	
	// Only ORT backend is supported for image-to-text
	if p.Runtime != "ORT" {
		return fmt.Errorf("image-to-text pipeline only supports ORT backend, got: %s", p.Runtime)
	}
	
	// Stage 1: Run encoder on image to get encoder hidden states
	inputTensors := batch.InputValues.([]ort.Value)
	encoderOutputs := make([]ort.Value, len(p.EncoderModel.OutputsMeta))
	
	err := p.EncoderModel.ORTModel.Session.Run(inputTensors, encoderOutputs)
	if err != nil {
		return fmt.Errorf("encoder forward pass failed: %w", err)
	}
	defer func() {
		for _, tensor := range encoderOutputs {
			if tensor != nil {
				tensor.Destroy()
			}
		}
	}()
	
	// Get encoder hidden states (typically the first output: last_hidden_state)
	encoderHiddenStates := encoderOutputs[0]
	
	// Stage 2: Autoregressive decoder loop
	generatedTokens := make([][]int64, batchSize)
	eosTokenIDs := p.DecoderModel.EosTokenIDs
	decoderStartTokenID := int64(p.DecoderModel.PadToken) // BOS/decoder_start_token
	
	// Initialize with decoder start token
	initialTokens := make([]int64, batchSize)
	for i := range initialTokens {
		initialTokens[i] = decoderStartTokenID
	}
	
	// Track which sequences have finished
	finish := make([]bool, batchSize)
	finishCount := 0
	
	// Create input_ids tensor for first step
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(batchSize64, 1), initialTokens)
	if err != nil {
		return fmt.Errorf("failed to create initial input_ids tensor: %w", err)
	}
	
	// Map decoder inputs by name for flexible input ordering
	inputMetaMap := make(map[string]int)
	for i, inputMeta := range p.DecoderModel.InputsMeta {
		inputMetaMap[inputMeta.Name] = i
	}
	
	// Autoregressive generation loop
	for step := 0; step < p.MaxNewTokens; step++ {
		// Prepare decoder inputs in correct order based on model metadata
		decoderInputs := make([]ort.Value, len(p.DecoderModel.InputsMeta))
		
		for i, inputMeta := range p.DecoderModel.InputsMeta {
			switch inputMeta.Name {
			case "input_ids":
				decoderInputs[i] = inputIDsTensor
			case "encoder_hidden_states", "encoder_outputs":
				decoderInputs[i] = encoderHiddenStates
			case "use_cache_branch":
				// This is a boolean control input for ONNX graph optimization
				// Set to false (0) for simplicity (not using KV cache)
				cacheBranch := []bool{false}
				cacheTensor, cacheErr := ort.NewTensor(ort.NewShape(1), cacheBranch)
				if cacheErr != nil {
					return fmt.Errorf("failed to create use_cache_branch tensor: %w", cacheErr)
				}
				defer cacheTensor.Destroy()
				decoderInputs[i] = cacheTensor
			default:
				return fmt.Errorf("unhandled decoder input: %s", inputMeta.Name)
			}
		}
		
		// Run decoder
		decoderOutputs := make([]ort.Value, len(p.DecoderModel.OutputsMeta))
		err = p.DecoderModel.ORTModel.Session.Run(decoderInputs, decoderOutputs)
		if err != nil {
			return fmt.Errorf("decoder forward pass failed at step %d: %w", step, err)
		}
		
		// Extract logits (first output)
		logitsTensor, ok := decoderOutputs[0].(*ort.Tensor[float32])
		if !ok {
			return fmt.Errorf("decoder output is not float32 tensor")
		}
		logits := logitsTensor.GetData()
		
		// Clean up decoder outputs
		for _, out := range decoderOutputs {
			if out != nil {
				out.Destroy()
			}
		}
		
		// Perform greedy sampling on last token logits
		// Logits shape: (batch_size, sequence_length, vocab_size)
		vocabSize := p.DecoderModel.VocabSize
		if vocabSize == 0 {
			return fmt.Errorf("decoder model vocab size is 0")
		}
		seqLen := len(logits) / (batchSize * vocabSize)
		greedyTokens := make([]int64, batchSize)
		
		for i := 0; i < batchSize; i++ {
			if !finish[i] {
				// Get logits for the last position of this sequence
				lastTokenStart := i*seqLen*vocabSize + (seqLen-1)*vocabSize
				
				// Find token with highest logit (greedy decoding)
				maxIdx := 0
				maxLogit := logits[lastTokenStart]
				for j := 1; j < vocabSize; j++ {
					if logits[lastTokenStart+j] > maxLogit {
						maxLogit = logits[lastTokenStart+j]
						maxIdx = j
					}
				}
				
				greedyTokens[i] = int64(maxIdx)
				generatedTokens[i] = append(generatedTokens[i], greedyTokens[i])
				
				// Check if this sequence hit EOS
				if eosTokenIDs[greedyTokens[i]] {
					finish[i] = true
					finishCount++
				}
			} else {
				// Sequence already finished, use PAD token
				greedyTokens[i] = decoderStartTokenID
			}
		}
		
		// Clean up old input_ids tensor
		inputIDsTensor.Destroy()
		
		// Check if all sequences finished
		if finishCount >= batchSize {
			break
		}
		
		// Prepare inputs for next iteration by appending new tokens
		newSeqLen := int64(step + 2) // BOS + generated tokens so far
		newInputIDs := make([]int64, batchSize*int(newSeqLen))
		
		for i := 0; i < batchSize; i++ {
			offset := i * int(newSeqLen)
			newInputIDs[offset] = decoderStartTokenID
			for j, token := range generatedTokens[i] {
				newInputIDs[offset+j+1] = token
			}
		}
		
		inputIDsTensor, err = ort.NewTensor(ort.NewShape(batchSize64, newSeqLen), newInputIDs)
		if err != nil {
			return fmt.Errorf("failed to create input_ids tensor for step %d: %w", step+1, err)
		}
	}
	
	// Final cleanup
	if inputIDsTensor != nil {
		inputIDsTensor.Destroy()
	}
	
	// Store generated tokens as output
	batch.OutputValues = make([]any, batchSize)
	for i := range generatedTokens {
		batch.OutputValues[i] = generatedTokens[i]
	}
	
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess converts model outputs to generated text.
// Decodes token IDs to human-readable text using the tokenizer.
func (p *ImageToTextPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*ImageToTextOutput, error) {
	outputValues := batch.OutputValues
	var batchResults []ImageToTextResult

	// Process each output in the batch
	for _, val := range outputValues {
		var tokenIDs []int64

		// Handle different output types
		switch v := val.(type) {
		case []int64:
			// Token IDs directly from model
			tokenIDs = v
		case []int32:
			// Convert int32 to int64
			tokenIDs = make([]int64, len(v))
			for i, tok := range v {
				tokenIDs[i] = int64(tok)
			}
		case [][]int64:
			// For models that return 2D arrays, take the first sequence
			if len(v) > 0 {
				tokenIDs = v[0]
			}
		case [][]int32:
			// For models that return 2D arrays of int32
			if len(v) > 0 {
				tokenIDs = make([]int64, len(v[0]))
				for i, tok := range v[0] {
					tokenIDs[i] = int64(tok)
				}
			}
		default:
			return nil, fmt.Errorf("output type %T is not supported", val)
		}

		// Convert int64 to uint32 for the decoder
		convertedTokens := make([]uint32, len(tokenIDs))
		for i, tok := range tokenIDs {
			convertedTokens[i] = uint32(tok)
		}

		// Decode tokens to text (skip_special_tokens=True)
		decodedString, err := pipelineBackends.Decode(convertedTokens, p.Model.Tokenizer, true)
		if err != nil {
			return nil, fmt.Errorf("error decoding generated tokens: %w", err)
		}

		batchResults = append(batchResults, ImageToTextResult{
			GeneratedText: decodedString,
		})
	}

	return &ImageToTextOutput{Results: batchResults}, nil
}

// Run runs the pipeline on a batch of image file paths.
func (p *ImageToTextPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline returns the concrete output type.
func (p *ImageToTextPipeline) RunPipeline(inputs []string) (*ImageToTextOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch(len(inputs))
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	images, err := util.LoadImagesFromPaths(inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}

	runErrors = append(runErrors, p.Preprocess(batch, images))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}

func (p *ImageToTextPipeline) RunWithImages(inputs []image.Image) (*ImageToTextOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch(len(inputs))
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
