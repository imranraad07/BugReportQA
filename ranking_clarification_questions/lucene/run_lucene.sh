SITE_DIR=$1

/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/javac -cp 'lucene-6.3.0/libs/*' IndexFiles.java

/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/javac -cp 'lucene-6.3.0/libs/*' SearchSimilarFiles.java

/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -cp '.:lucene-6.3.0/libs/*' IndexFiles -docs $SITE_DIR/post_docs/ -index $SITE_DIR/post_doc_indices/

/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -cp '.:lucene-6.3.0/libs/*' SearchSimilarFiles -index $SITE_DIR/post_doc_indices/ -queries $SITE_DIR/post_docs/ -outputFile $SITE_DIR/lucene_similar_posts.txt
