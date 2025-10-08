using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;

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

    [Header("Postprocessing Rules")]
    public bool enforceGroundRow = false; // force bottom row '#'
    public bool enforceRules = false;     // pipes stack + must have ground; '?' needs sky below; 'c' sits on ground
    public bool solidGround = false;      // gaps are empty all the way down; no floating ground
    [Range(0f, 0.5f)] public float gapRate = 0.0f; // probability to start a gap on bottom row
    public int gapMin = 2;
    public int gapMax = 6;
    public int postprocessSeed = 12345;

    [Header("Placement")]
    public LevelGenerator levelGenerator; // Optional: reference a level placer in your scene
    [Header("Saving")]
    public bool saveOnGenerate = false;
    public string savePath = @"C:\\Users\\User\\Documents\\GitHub\\Machine-Learning---Mario-Levels\\generated_level_unity.txt";

    private IWorker worker;
    private string[] itos;                   // id -> char
    private Dictionary<string, int> stoi;    // char -> id
    private string lastGeneratedText;        // cached text
    private string[] lastGeneratedLines;     // cached shaped lines

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
        if (onnxModel == null)
        {
            Debug.LogError("BarracudaLstmGenerator: onnxModel is not assigned");
            return;
        }
        if (itosText == null)
        {
            Debug.LogError("BarracudaLstmGenerator: itosText is not assigned");
            return;
        }
        var txt = Generate(length, seedText);
        // Expand compound token: § -> pP
        txt = txt.Replace("§", "pP");
        var lines = WrapToLines(txt, wrapWidth, targetHeight);
        // Apply postprocessing rules similar to the Python path
        lines = ApplyPostprocessing(lines);
        lastGeneratedText = string.Join("\n", lines);
        lastGeneratedLines = lines;
        Debug.Log($"Generator: textLen={txt?.Length ?? 0}, lines={lines?.Length ?? 0}, width={(lines!=null && lines.Length>0 ? lines[0].Length : 0)}");
        if (levelGenerator != null)
        {
            levelGenerator.BuildLevel(lines);
        }
        else
        {
            Debug.LogWarning("LevelGenerator is not assigned; generation text produced but not placed.");
        }
        if (saveOnGenerate)
        {
            try { SaveLastLevelToFile(string.IsNullOrWhiteSpace(savePath) ? null : savePath); }
            catch (System.Exception ex) { Debug.LogError($"SaveLastLevelToFile failed: {ex.Message}"); }
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
        if (lines.Count == 0) lines.Add(new string('-', Math.Max(1, width)));
        if (height > 0)
        {
            if (lines.Count > height) lines = lines.GetRange(0, height);
            while (lines.Count < height) lines.Add(new string('-', width));
        }
        return lines.ToArray();
    }

    // Save the last generated, postprocessed text to a file
    public void SaveLastLevelToFile(string path = "")
    {
        if (lastGeneratedLines == null || lastGeneratedLines.Length == 0)
        {
            Debug.LogWarning("No generated level to save. Call GenerateAndBuild first.");
            return;
        }
        if (string.IsNullOrEmpty(path))
        {
            var dir = Application.persistentDataPath;
            path = Path.Combine(dir, "generated_level_unity.txt");
        }
        var dirPath = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dirPath)) Directory.CreateDirectory(dirPath);
        File.WriteAllLines(path, lastGeneratedLines);
        Debug.Log($"Saved level text to {path}");
    }

    string[] ApplyPostprocessing(string[] lines)
    {
        if (lines == null || lines.Length == 0) return lines;
        int h = targetHeight > 0 ? targetHeight : lines.Length;
        int w = wrapWidth > 0 ? wrapWidth : (lines.Length > 0 ? lines[0].Length : 0);
        if (h <= 0 || w <= 0) return lines;

        // Build rectangular grid
        var grid = new char[h, w];
        for (int y = 0; y < h; y++)
        {
            var ln = y < lines.Length ? lines[y] : string.Empty;
            for (int x = 0; x < w; x++)
            {
                grid[y, x] = x < ln.Length ? ln[x] : '-';
            }
        }

        if (enforceGroundRow)
        {
            for (int x = 0; x < w; x++) grid[h - 1, x] = '#';
        }

        if (enforceRules)
        {
            EnforcePipesAndSupports(grid, h, w);
            EnforceQuestionBlocks(grid, h, w);
            EnforceCannons(grid, h, w);
        }

        if (gapRate > 0f)
        {
            CarveGroundGaps(grid, h, w, gapRate, Mathf.Max(1, gapMin), Mathf.Max(gapMin, gapMax), postprocessSeed);
        }

        if (solidGround)
        {
            EnforceSolidGround(grid, h, w);
        }

        // Back to lines
        var outLines = new string[h];
        for (int y = 0; y < h; y++)
        {
            var chars = new char[w];
            for (int x = 0; x < w; x++) chars[x] = grid[y, x];
            outLines[y] = new string(chars);
        }
        return outLines;
    }

    static bool IsSky(char c) => c == '-' || c == ' ';
    static bool IsPipe(char c) => c == 'p' || c == 'P';

    void EnforcePipesAndSupports(char[,] grid, int h, int w)
    {
        // Fill sky under pipes with 'P', add ground if reaches bottom
        for (int y = 0; y < h - 1; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var ch = grid[y, x];
                if (IsPipe(ch))
                {
                    int yb = y + 1;
                    while (yb < h && IsSky(grid[yb, x]))
                    {
                        grid[yb, x] = 'P';
                        yb++;
                    }
                    if (yb >= h)
                    {
                        grid[h - 1, x] = '#';
                    }
                }
            }
        }
        // Ensure vertical continuity within any pipe column
        for (int x = 0; x < w; x++)
        {
            int top = -1, bot = -1;
            for (int y = 0; y < h; y++) if (IsPipe(grid[y, x])) { top = y; break; }
            for (int y = h - 1; y >= 0; y--) if (IsPipe(grid[y, x])) { bot = y; break; }
            if (top >= 0 && bot >= top)
            {
                for (int y = top; y <= bot; y++) if (IsSky(grid[y, x])) grid[y, x] = 'P';
                if (bot < h - 1 && grid[bot + 1, x] != '#') grid[bot + 1, x] = '#';
            }
        }
    }

    void EnforceQuestionBlocks(char[,] grid, int h, int w)
    {
        for (int y = 0; y < h - 1; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (grid[y, x] == '?')
                {
                    if (!IsSky(grid[y + 1, x]))
                    {
                        // Prefer move up if sky above
                        if (y > 0 && IsSky(grid[y - 1, x]))
                        {
                            grid[y - 1, x] = '?';
                            grid[y, x] = '-';
                        }
                        else if (grid[y + 1, x] != '#' && !IsPipe(grid[y + 1, x]) && grid[y + 1, x] != 'c')
                        {
                            grid[y + 1, x] = '-';
                        }
                    }
                }
            }
        }
    }

    void EnforceCannons(char[,] grid, int h, int w)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (grid[y, x] == 'c')
                {
                    if (y < h - 1 && grid[y + 1, x] != '#') grid[y + 1, x] = '#';
                }
            }
        }
    }

    void CarveGroundGaps(char[,] grid, int h, int w, float rate, int minW, int maxW, int seed)
    {
        var protectedCols = new HashSet<int>();
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h - 1; y++)
            {
                var c = grid[y, x];
                if (IsPipe(c) || c == 'c') { protectedCols.Add(x); break; }
            }
        }
        var rng = new System.Random(seed);
        int yb = h - 1;
        int x0 = 0;
        while (x0 < w)
        {
            if (grid[yb, x0] == '#' && !protectedCols.Contains(x0) && rng.NextDouble() < rate)
            {
                int gapW = rng.Next(minW, maxW + 1);
                int end = Math.Min(w, x0 + gapW);
                for (int x = x0; x < end; x++)
                {
                    if (protectedCols.Contains(x)) break;
                    grid[yb, x] = '-';
                }
                x0 = end + 1; // ensure at least one ground after a gap
            }
            else x0++;
        }
    }

    void EnforceSolidGround(char[,] grid, int h, int w)
    {
        for (int x = 0; x < w; x++)
        {
            bool bottomIsGround = grid[h - 1, x] == '#';
            if (bottomIsGround)
            {
                int topGround = h - 1;
                for (int y = 0; y < h; y++) if (grid[y, x] == '#') { topGround = y; break; }
                for (int y = 0; y < h; y++)
                {
                    if (y >= topGround) { if (IsSky(grid[y, x])) grid[y, x] = '#'; }
                    else { if (grid[y, x] == '#') grid[y, x] = '-'; }
                }
            }
            else
            {
                for (int y = 0; y < h; y++) if (grid[y, x] == '#') grid[y, x] = '-';
            }
        }
    }
}

