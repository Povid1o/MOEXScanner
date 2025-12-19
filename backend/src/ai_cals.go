package src

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

type DeepSeekAnswer struct {
	ID       string    `json:"id"`
	Provider string    `json:"provider"`
	Model    string    `json:"model"`
	Object   string    `json:"object"`
	Created  int64     `json:"created"`
	Choices  []Choices `json:"choices"`
}

type Choices struct {
	Logprobs             string  `json:"logprobs"`
	Finisd_reason        string  `json:"finisd_reason"`
	Native_finish_reason string  `json:"native_finish_reason"`
	Stop                 string  `json:"stop"`
	Message              Message `json:"message"`
}
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func CheckError(err error) {
	if err != nil {
		log.Print("[[ai_cals]]: ", err)

	}
}

func Ai_send_request(role string, text string) (string, error) { // if !role {role = "user"}
	url := "https://openrouter.ai/api/v1/chat/completions"
	payload := map[string]interface{}{
		"model": "tngtech/deepseek-r1t2-chimera:free",
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": text,
			},
		},
	}

	// Сериализация в JSON
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal JSON: %w", err)
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", API_KEY))

	if err != nil {
		fmt.Println("Error creating request:", err)
		return "", err
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Error sending request:", err)
		return "", err
	}

	defer resp.Body.Close()

	CheckError(err)
	result, err := io.ReadAll(resp.Body)
	if err != nil {
		CheckError(err)
	}
	var apiResponse DeepSeekAnswer
	if err := json.Unmarshal(result, &apiResponse); err != nil {
		log.Print("JSON parse error:", err)
	}

	log.Print("apiResponse", apiResponse)
	if len(apiResponse.Choices) > 0 {
		content := apiResponse.Choices[0].Message.Content
		if content != "" {
			return content, nil
		}
	}

	log.Print("[ai]: Что-то с AI не так")
	return "False", fmt.Errorf("ai errror")
}

// Ai_send_request_local runs the local Python CLI that performs prediction.
// It executes: python3 ../ML/scripts/predict_cli.py and pipes JSON to stdin.
func Ai_send_request_local(payload map[string]interface{}) (string, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %w", err)
	}

	// model server URL from env or default
	url := os.Getenv("ML_MODEL_URL")
	if url == "" {
		url = "http://127.0.0.1:8000/predict_local"
	}

	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= 400 {
		return string(body), fmt.Errorf("model server returned status %d", resp.StatusCode)
	}

	return string(body), nil
}
