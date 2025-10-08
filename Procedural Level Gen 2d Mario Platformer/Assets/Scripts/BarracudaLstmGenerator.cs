using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class BarracudaLstmGenerator : MonoBehaviour
{
    [Header("Model & Vocab")]
    public NNModel onnxModel;        // Drag mario_lstm.onnx asset here
    public TextAsset itosText;       // Resources/itos.txt
    public int hiddenDim = 256;      // Must match export
    public int numLayers = 2;        // Must match export

    [Header("Sampling")]
    [Range(0.1f, 2.0f)] public float temperature = 0.9f;
    public int topK = 0;
    [Range(0f, 1f)] public float topP = 0f;
    public float hashBias = 0.0f;    // Boosts '#' logit
    public string[] bannedTokens = new[] { "|" };

    [Header("Shaping")]
    public int wrapWidth = 120;
    public int targetHeight = 14;

    [Header("Placement")]
    public LevelGenerator levelGenerator; // Optional: reference a level placer in your scene

    private IWorker worker;
    private string[] itos;                   // id -> char
    private Dictionary<string, int> stoi;    // char -> id

    void Awake()
    {
        var model = ModelLoader.Load(onnxModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
        LoadVocab();
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }

    void LoadVocab()
    {
        var lines = itosText.text.Split(new[] {"\r\n", "\n", "\r"}, StringSplitOptions.None);
        itos = lines.ToArray();
        stoi = new Dictionary<string, int>();
        for (int i = 0; i < itos.Length; i++)
        {
            var ch = itos[i];
            if (ch == "\\n") ch = "\n"; // convert literal back to newline
            stoi[ch] = i;
        }
        if (!stoi.ContainsKey("\n") && itos.Length > 0)
            stoi["\n"] = 0;
    }

    public void GenerateAndBuild(int length = 1680, string seedText = "")
    {
        var txt = Generate(length, seedText);
        // Expand compound token: § -> pP
        txt = txt.Replace("§", "pP");
        var lines = WrapToLines(txt, wrapWidth, targetHeight);
        if (levelGenerator != null)
        {
            levelGenerator.BuildLevel(lines);
        }
        else
        {
            Debug.LogWarning("LevelGenerator is not assigned; generation text produced but not placed.");
        }
    }

    string Generate(int length, string seedText)
    {
        int vocab = itos.Length;
        var bannedIds = new HashSet<int>(bannedTokens.Where(t => stoi.ContainsKey(t)).Select(t => stoi[t]));
        int hashId = stoi.ContainsKey("#") ? stoi["#"] : -1;

        // init hidden state tensors
        // Create zeroed hidden state tensors with shape [numLayers,1,hiddenDim]
        var hShape = new TensorShape(numLayers, 1, hiddenDim, 1);
        var cShape = new TensorShape(numLayers, 1, hiddenDim, 1);
        var h0 = new Tensor(hShape, new float[hShape.length]);
        var c0 = new Tensor(cShape, new float[cShape.length]);

        List<int> outIds = new List<int>();
        int lastId = stoi.ContainsKey("\n") ? stoi["\n"] : 0;

        // warm-up with seed
        foreach (var ch in seedText)
        {
            var s = ch.ToString();
            var cid = stoi.ContainsKey(s) ? stoi[s] : lastId;
            Step(cid, ref h0, ref c0);
            outIds.Add(cid);
            lastId = cid;
        }

        for (int i = 0; i < Math.Max(0, length); i++)
        {
            int nextId = SampleNext(lastId, h0, c0, bannedIds, hashId);
            Step(nextId, ref h0, ref c0);
            outIds.Add(nextId);
            lastId = nextId;
        }

        var chars = outIds.Select(id => itos[id] == "\\n" ? "\n" : itos[id]);
        return string.Concat(chars);
    }

    void Step(int tokenId, ref Tensor h, ref Tensor c)
    {
        using var x = new Tensor(new TensorShape(1, 1), new float[] { (float)tokenId });
        var inputs = new Dictionary<string, Tensor> { { "x", x }, { "h0", h }, { "c0", c } };
        worker.Execute(inputs);
        using var hn = worker.PeekOutput("hn").DeepCopy();
        using var cn = worker.PeekOutput("cn").DeepCopy();
        h.Dispose(); c.Dispose();
        h = hn; c = cn;
    }

    int SampleNext(int lastId, Tensor h, Tensor c, HashSet<int> banned, int hashId)
    {
        using var x = new Tensor(new TensorShape(1, 1), new float[] { (float)lastId });
        var inputs = new Dictionary<string, Tensor> { { "x", x }, { "h0", h }, { "c0", c } };
        worker.Execute(inputs);
        using var logitsT = worker.PeekOutput("logits");

        int vocab = itos.Length;
        float[] logits = new float[vocab];
        var flat = logitsT.ToReadOnlyArray();
        for (int i = 0; i < vocab && i < flat.Length; i++) logits[i] = flat[i];

        // temperature
        for (int i = 0; i < vocab; i++) logits[i] /= Mathf.Max(1e-6f, temperature);
        // hash bias
        if (hashId >= 0 && Mathf.Abs(hashBias) > 0f) logits[hashId] += hashBias;
        // ban tokens
        foreach (var b in banned) logits[b] = float.NegativeInfinity;

        // top-k
        if (topK > 0 && topK < vocab)
        {
            int[] idx = Enumerable.Range(0, vocab).ToArray();
            Array.Sort(idx, (a, b) => logits[b].CompareTo(logits[a]));
            var keep = new bool[vocab];
            for (int i = 0; i < topK; i++) keep[idx[i]] = true;
            for (int i = 0; i < vocab; i++) if (!keep[i]) logits[i] = float.NegativeInfinity;
        }
        // top-p
        if (topP > 0f && topP < 1f)
        {
            int[] idx = Enumerable.Range(0, vocab).ToArray();
            Array.Sort(idx, (a, b) => logits[b].CompareTo(logits[a]));
            var probs = Softmax(logits);
            float cum = 0f;
            var allow = new bool[vocab];
            for (int i = 0; i < idx.Length; i++)
            {
                cum += probs[idx[i]];
                allow[idx[i]] = true;
                if (cum >= topP) break;
            }
            for (int i = 0; i < vocab; i++) if (!allow[i]) logits[i] = float.NegativeInfinity;
        }

        var finalProbs = Softmax(logits);
        return SampleFromDistribution(finalProbs);
    }

    static float[] Softmax(float[] logits)
    {
        float max = logits.Where(v => !float.IsNegativeInfinity(v)).DefaultIfEmpty(0f).Max();
        double sum = 0.0;
        var exps = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNegativeInfinity(logits[i])) { exps[i] = 0; continue; }
            exps[i] = Math.Exp(logits[i] - max);
            sum += exps[i];
        }
        var probs = new float[logits.Length];
        if (sum <= 0)
        {
            float u = 1f / logits.Length;
            for (int i = 0; i < probs.Length; i++) probs[i] = u;
            return probs;
        }
        for (int i = 0; i < probs.Length; i++) probs[i] = (float)(exps[i] / sum);
        return probs;
    }

    static int SampleFromDistribution(float[] probs)
    {
        float r = UnityEngine.Random.value;
        float cum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.Length - 1;
    }

    static string[] WrapToLines(string text, int width, int height)
    {
        if (width <= 0) width = 120;
        var lines = new List<string>();
        int i = 0;
        while (i < text.Length)
        {
            int len = Math.Min(width, text.Length - i);
            lines.Add(text.Substring(i, len));
            i += len;
        }
        if (height > 0)
        {
            if (lines.Count > height) lines = lines.GetRange(0, height);
            while (lines.Count < height) lines.Add(new string('-', width));
        }
        return lines.ToArray();
    }
}

