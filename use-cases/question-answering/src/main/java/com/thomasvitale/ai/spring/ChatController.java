package com.thomasvitale.ai.spring;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
class ChatController {
    private static final Logger logger = LoggerFactory.getLogger(ChatController.class);
    
    private final ChatClient chatClient;
    private final RetrievalAugmentationAdvisor retrievalAugmentationAdvisor;

    ChatController(
            ChatClient.Builder chatClientBuilder, 
            VectorStore vectorStore,
            @Value("${rag.max-results:5}") int maxResults,
            @Value("${rag.similarity-threshold:0.7}") double similarityThreshold) {
        
        this.chatClient = chatClientBuilder.clone().build();
        this.retrievalAugmentationAdvisor = RetrievalAugmentationAdvisor.builder()
                .documentRetriever(VectorStoreDocumentRetriever.builder()
                        .vectorStore(vectorStore)
                        .similarityThreshold(similarityThreshold)
                        .topK(maxResults)
                        .build())
                .build();
    }

    @PostMapping("/chat/doc")
    String chatWithDocument(@Valid @RequestBody ChatRequest request) {
        logger.info("Traitement de la question: {}", request.question());
        long startTime = System.currentTimeMillis();
        
        try {
            String response = chatClient.prompt()
                    .advisors(retrievalAugmentationAdvisor)
                    .user(request.question())
                    .call()
                    .content();
            
            long duration = System.currentTimeMillis() - startTime;
            logger.info("Réponse générée en {} ms", duration);
            return response;
            
        } catch (Exception e) {
            logger.error("Erreur lors du traitement", e);
            throw new RuntimeException("Erreur lors du traitement de la question", e);
        }
    }

    @PostMapping("/chat/doc/stream")
    Flux<String> chatWithDocumentStream(@Valid @RequestBody ChatRequest request) {
        return chatClient.prompt()
                .advisors(retrievalAugmentationAdvisor)
                .user(request.question())
                .stream()
                .content()
                .doOnSubscribe(subscription -> 
                    logger.info("Streaming démarré pour: {}", request.question()))
                .doOnError(error -> 
                    logger.error("Erreur dans le streaming", error));
    }
}

record ChatRequest(
    @NotBlank(message = "La question est obligatoire")
    String question,
    String sessionId,
    Double temperature
) {}
