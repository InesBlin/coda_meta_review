for value in regular var_mod study_mod
do
    # # # # Learning
    java -Xmx12G -cp AnyBURL-23-1.jar de.unima.ki.anyburl.Learn coda/"$value"/config-learn.properties

    # # # # Predicting
    java -Xmx12G -cp AnyBURL-23-1.jar de.unima.ki.anyburl.Apply coda/"$value"/config-apply.properties

    # # # # Copy file for readable outputs
    cp coda/"$value"/preds-10 coda/"$value"/h_"$value"_es_d

done