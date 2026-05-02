using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Sends an HTTP GET request to the local TOAD-GAN Flask server and
/// delivers the raw JSON string to any registered listener.
///
/// Usage
/// -----
/// 1. Attach this component to any persistent GameObject (e.g. GameManager).
/// 2. Call <see cref="RequestLevel"/> to kick off a generation.
/// 3. Subscribe to <see cref="OnLevelReceived"/> to consume the result.
///
/// The Flask server must be running at <see cref="serverUrl"/> before
/// Unity tries to fetch a level.
/// </summary>
public class ApiClient : MonoBehaviour
{
    // ── Inspector settings ────────────────────────────────────────────────

    [Header("Server Settings")]
    [Tooltip("Base URL of the running TOAD-GAN Flask server.")]
    public string serverUrl = "http://localhost:5000";

    [Tooltip("How many seconds to wait before treating the request as failed.")]
    public int timeoutSeconds = 60;

    // ── Events ────────────────────────────────────────────────────────────

    /// <summary>Raised when a level JSON string is successfully received.</summary>
    public event Action<string> OnLevelReceived;

    /// <summary>Raised when the request fails. Argument is the error message.</summary>
    public event Action<string> OnError;

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Request a new level from the server with optional generation parameters.
    /// </summary>
    /// <param name="temperature">Noise temperature – higher = more variation.</param>
    /// <param name="scaleW">Width multiplier relative to the training level size.</param>
    public void RequestLevel(float temperature = 1.0f, float scaleW = 1.0f)
    {
        string url = $"{serverUrl}/generate?temperature={temperature}&scale_w={scaleW}";
        StartCoroutine(FetchLevel(url));
    }

    // ── Private ───────────────────────────────────────────────────────────

    private IEnumerator FetchLevel(string url)
    {
        Debug.Log($"[ApiClient] Requesting level from: {url}");

        using UnityWebRequest request = UnityWebRequest.Get(url);
        request.timeout = timeoutSeconds;

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string json = request.downloadHandler.text;
            Debug.Log($"[ApiClient] Level received ({json.Length} chars).");
            OnLevelReceived?.Invoke(json);
        }
        else
        {
            string err = $"[ApiClient] Request failed: {request.error} (HTTP {request.responseCode})";
            Debug.LogError(err);
            OnError?.Invoke(err);
        }
    }
}
