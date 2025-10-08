using UnityEngine;

public class GenerateOnStart : MonoBehaviour
{
    public BarracudaLstmGenerator generator;
    public int length = 240;
    public string seedText = "";

    void Start()
    {
        if (generator == null)
        {
            generator = GetComponent<BarracudaLstmGenerator>();
        }
        if (generator == null)
        {
            Debug.LogError("GenerateOnStart: missing BarracudaLstmGenerator reference");
            return;
        }
        generator.GenerateAndBuild(length, seedText);
        Debug.Log("GenerateOnStart: level generated");
    }
}