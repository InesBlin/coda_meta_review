for value in regular var_mod study_mod
do

    # Evaluating
    printf "HYPOTHESIS TYPE\t$value\n"
    java -Xmx12G -cp AnyBURL-23-1.jar de.unima.ki.anyburl.Eval coda/"$value"/config-eval.properties
    printf "=======\n"
done