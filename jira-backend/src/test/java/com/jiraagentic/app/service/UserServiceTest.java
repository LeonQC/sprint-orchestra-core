package com.jiraagentic.app.service;

import com.jiraagentic.app.dto.CreateUserRequest;
import com.jiraagentic.app.entity.User;
import com.jiraagentic.app.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    // Low cost factor keeps the test fast; production uses the default (10).
    private final PasswordEncoder encoder = new BCryptPasswordEncoder(4);

    @Mock UserRepository userRepository;

    private UserService service() {
        return new UserService(userRepository, encoder);
    }

    private User userWithPassword(String stored) {
        User user = new User();
        user.setUsername("alice");
        user.setPassword(stored);
        return user;
    }

    @Test
    void acceptsTheRightPasswordAgainstItsHash() {
        when(userRepository.findByUsername("alice")).thenReturn(Optional.of(userWithPassword(encoder.encode("123"))));
        assertThat(service().authenticate("alice", "123").getUsername()).isEqualTo("alice");
    }

    @Test
    void acceptsAHashWrittenByPgcrypto() {
        // The V35 migration and the seed scripts hash with Postgres crypt(); same "123".
        String pgcryptoHash = "$2a$10$.7G43/qDfW6zKRPSphqVg.eF9SXvKTV9HLmlrSOsa48.W1sFXqYoi";
        when(userRepository.findByUsername("alice")).thenReturn(Optional.of(userWithPassword(pgcryptoHash)));
        assertThat(service().authenticate("alice", "123").getUsername()).isEqualTo("alice");
    }

    @Test
    void rejectsAWrongPassword() {
        when(userRepository.findByUsername("alice")).thenReturn(Optional.of(userWithPassword(encoder.encode("123"))));
        assertThatThrownBy(() -> service().authenticate("alice", "1234"))
                .hasMessage("Invalid username or password");
    }

    @Test
    void unknownUserGetsTheSameMessageAsAWrongPassword() {
        when(userRepository.findByUsername("nobody")).thenReturn(Optional.empty());
        assertThatThrownBy(() -> service().authenticate("nobody", "123"))
                .hasMessage("Invalid username or password");
    }

    @Test
    void aLeftoverPlainTextPasswordNoLongerWorks() {
        when(userRepository.findByUsername("alice")).thenReturn(Optional.of(userWithPassword("123")));
        assertThatThrownBy(() -> service().authenticate("alice", "123"))
                .hasMessage("Invalid username or password");
    }

    @Test
    void createdUsersAreStoredWithAHash() {
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> invocation.getArgument(0));
        CreateUserRequest req = new CreateUserRequest();
        req.setUsername("bob");
        User saved = service().create(req);
        assertThat(saved.getPassword()).startsWith("$2a$").isNotEqualTo("123");
        assertThat(encoder.matches("123", saved.getPassword())).isTrue();
    }
}
