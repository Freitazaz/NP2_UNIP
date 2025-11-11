#include "ia_client.h"
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

struct Memory {
    char *response;
    size_t size;
};

static size_t write_callback(void *data, size_t size, size_t nmemb, void *userp) {
    size_t total = size * nmemb;
    struct Memory *mem = (struct Memory *)userp;
    char *ptr = realloc(mem->response, mem->size + total + 1);
    if(ptr == NULL) return 0;
    mem->response = ptr;
    memcpy(&(mem->response[mem->size]), data, total);
    mem->size += total;
    mem->response[mem->size] = 0;
    return total;
}

void ia_init(IA_Config *cfg, const char *url, int timeout) {
    strncpy(cfg->base_url, url, sizeof(cfg->base_url));
    cfg->timeout = timeout;
}

int ia_search(IA_Config *cfg, const char *query, int limit, char *out, size_t out_size) {
    CURL *curl;
    CURLcode res;
    struct Memory chunk = {0};

    char url[512];
    snprintf(url, sizeof(url), "%s/search?q=%s&limit=%d", cfg->base_url, query, limit);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, cfg->timeout);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);

        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "Erro na requisição: %s\n", curl_easy_strerror(res));
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            return -1;
        }

        strncpy(out, chunk.response, out_size - 1);
        out[out_size - 1] = '\0';

        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    return 0;
}