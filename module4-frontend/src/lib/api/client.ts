class GeoChatAPIClient {
  /**
   * API client for backend integration
   * Measurable Success: <300ms API response time, 99% success rate
   */
  
  private baseURL: string;
  private authToken: string | null;
  
  constructor(config: APIConfig) {
    this.baseURL = config.baseURL;
    this.authToken = config.authToken;
  }
  
  async sendChatMessage(sessionId: string, message: string): Promise<ChatResponse> {
    // Send chat message with error handling and retry logic
    // Success Metric: <2s end-to-end chat response time
  }
  
  async querySpatialData(queryParams: SpatialQueryParams): Promise<GeologicalRecord[]> {
    // Query geological data with spatial filters
    // Success Metric: <500ms spatial data retrieval
  }
  
  async authenticateUser(credentials: UserCredentials): Promise<AuthResponse> {
    // User authentication with token management
    // Success Metric: <300ms authentication response
  }
  
  private async handleAPIError(error: APIError): Promise<void> {
    // Comprehensive error handling with user-friendly messages
  }
  
  private async retryRequest<T>(request: () => Promise<T>, maxRetries: number = 3): Promise<T> {
    // Intelligent retry logic for failed requests
  }
}

class WebSocketManager {
  /**
   * WebSocket connection management for real-time features
   * Measurable Success: 99% connection uptime, <100ms message delivery
   */
  
  private socket: WebSocket | null;
  private reconnectAttempts: number;
  private maxReconnectAttempts: number;
  
  connect(sessionId: string): Promise<void> {
    // Establish WebSocket connection with auto-reconnection
    // Success Metric: <2s connection establishment
  }
  
  sendMessage(message: ChatMessage): void {
    // Send message with delivery confirmation
    // Success Metric: <100ms message delivery confirmation
  }
  
  onMessage(handler: (message: ChatResponse) => void): void {
    // Message event handling with type safety
  }
  
  private handleConnectionLoss(): void {
    // Automatic reconnection with exponential backoff
  }
  
  private validateConnection(): boolean {
    // Connection health monitoring
  }
}
